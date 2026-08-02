
"""Combined Image encoder and Text decoder Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.image_encoder.model import CTCEncoderConfig, ImageEncoderCTC
from src.text_decoder.model import GPTModel, GPTConfig, MultiHeadAttention, SwiGLU, calculate_positional_encodings
from transformers.modeling_outputs import CausalLMOutput
from src.text_decoder.grapheme_tokenizer.tokenizer import TeluguGraphemeTokenizer



class DecoderTransformerBlock(nn.Module):
    """Transformer block; cross-attention (with a zero-init tanh gate) is optional."""
    def __init__(self, config: GPTConfig, use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, dropout=config.dropout)

        if self.use_cross_attention:
            self.cross_attention_layer = MultiHeadAttention(config.embed_dim, config.num_heads, dropout=config.dropout)
            self.layer_norm1_5 = nn.LayerNorm(config.embed_dim)
            # Zero-init tanh gate (Flamingo-style): tanh(0) == 0, so the cross-attention
            # branch contributes nothing at init and is phased in as the gate trains.
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            SwiGLU(config.embed_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, attn_mask = None, padding_mask = None):
        normed = self.layer_norm1(x)
        attn_out = self.attention_layer(normed, normed, normed, attn_mask)
        x = x + attn_out

        if self.use_cross_attention:
            cross_in = self.layer_norm1_5(x)
            cross_out = self.cross_attention_layer(cross_in, encoder_output, encoder_output, padding_mask)
            x = x + torch.tanh(self.cross_attn_gate) * cross_out

        mlp_in = self.layer_norm2(x)
        mlp_out = self.mlp(mlp_in)
        x = x + mlp_out

        return x





class TextDecoder(nn.Module):
    """A text decoder model of the transformer model"""
    _keys_to_ignore_on_save = None
    def __init__(self, config: GPTConfig, pad_index: int):
        super().__init__()
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        self.register_buffer("positional_encodings", calculate_positional_encodings(torch.arange(config.ctx_len), config.embed_dim))
        # Cross-attention lives on even-indexed blocks only (every 2nd block).
        self.transformer_blocks = nn.ModuleList([
            DecoderTransformerBlock(config, use_cross_attention=(i % 2 == 0))
            for i in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear( config.embed_dim, config.vocab_size, bias=False)


    def _build_causal_attn_mask(self, input_ids, padding_mask):
        """
        Builds a combined boolean causal + padding mask.
        SDPA expects: True = attend, False = ignore.
        Shape: (B, 1, T, T)
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Causal mask: upper triangle is False (masked), lower triangle True
        causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()  # (T, T)
        if padding_mask is not None:
            # padding_mask: (B, T), 1=real token, 0=pad
            # Expand to (B, 1, 1, T) so it broadcasts over query positions
            pad_mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)      # (B, 1, 1, T)
            combined = causal.unsqueeze(0).unsqueeze(0) & pad_mask          # (B, 1, T, T)
        else:
            combined = causal.unsqueeze(0).unsqueeze(0)                     # (1, 1, T, T)
        return combined

    def forward(self, input_ids, encoder_output, text_padding_mask=None, img_text_padding_mask = None, labels = None):
        # input_ids -> (B, T)
        embeds = self.embedding_layer(input_ids)
        T = input_ids.shape[1]
        embeds = embeds + self.positional_encodings[:T].unsqueeze(0)
        causal_attn_mask = self._build_causal_attn_mask(input_ids, text_padding_mask)
        for block in self.transformer_blocks:
            embeds = block(embeds, encoder_output, attn_mask = causal_attn_mask, padding_mask = img_text_padding_mask)
        embeds = self.layer_norm(embeds)
        logits = self.lm_head(embeds)
        loss = None

        # Always calculate the loss
        # shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_index
        )
        return CausalLMOutput(
            loss=loss,
            logits=logits
        )


class EncoderDecoder(nn.Module):
    """An Image encoder and text decoder based transformer model"""
    def __init__(self, encoder_config: CTCEncoderConfig, decoder_config: GPTConfig, pad_index: int):
        super().__init__()
        self.encoder_model = ImageEncoderCTC(encoder_config)
        self.decoder_model = TextDecoder(decoder_config, pad_index)
        # Bridge the encoder width (384) to the decoder width (512) so the frame
        # features can feed the decoder's cross-attention keys/values. Trainable
        # (the encoder is frozen); a no-op nn.Identity when the widths already match.
        if encoder_config.embed_dim != decoder_config.embed_dim:
            self.enc_to_dec = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        else:
            self.enc_to_dec = nn.Identity()

    def forward(self, pixel_values, input_ids, input_lengths = None, text_padding_mask=None):
        # pixel_values -> (B, 1, H, W); input_lengths -> (B,) valid frames = W_real // downsample
        # encode() returns frame features (B, T, D) and a frame padding mask (True = padded frame).
        encoder_output, key_padding_mask = self.encoder_model.encode(pixel_values, input_lengths)
        encoder_output = self.enc_to_dec(encoder_output)         # (B, T, dec_embed_dim)

        if key_padding_mask is not None:
            # Custom cross-attention expects an SDPA-style mask (True = KEEP), shape (B, 1, 1, T).
            cross_key_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
        else:
            cross_key_mask = None
        decoder_output = self.decoder_model(input_ids, encoder_output, text_padding_mask, cross_key_mask, None)
        return decoder_output





if __name__ == '__main__':

    tokenizer = TeluguGraphemeTokenizer(vocab_file="/Users/xai/Personal/Projects/TeluguOCR/src/text_decoder/grapheme_tokenizer/telugu-vocab.json")
    print(tokenizer.pad_token_id)


    # -------------- Step-1: Load the pretrained models --------------
    # Pretrained text decoder model
    pretrained_config = GPTConfig(
        vocab_size=len(tokenizer),
        embed_dim=512,
        hidden_dim=1368,  # 2.67 * 512 = 2/3 * 4 * hidden_dim
        num_heads=8,
        num_layers=16,
        ctx_len=256,
        dropout=0.1
    )

    pretrained_text_model = GPTModel(pretrained_config, pad_index=tokenizer.pad_token_id)
    pretrained_text_model.load_state_dict(
        torch.load("/Users/xai/Personal/Projects/TeluguOCR/models/text_decoder/telugu-grapheme-gpt/final_model.pt", map_location="cpu")
    )

    # Load the pretrained image model
    img_encoder_cfg = CTCEncoderConfig(
        max_image_width=2048,
        max_frames=256
    )

    pretrained_encoder_model = ImageEncoderCTC(img_encoder_cfg)
    pretrained_encoder_model.load_state_dict(
        torch.load(
            '/Users/xai/Personal/Projects/TeluguOCR/models/image_encoder/ctc_encoder_stage-3/ctc-encoder-2048/final_model.pt',
            map_location="cpu")
    )

    # -------------- Step-2: Initialize the new encoder-decoder model --------------
    model = EncoderDecoder(
        encoder_config=img_encoder_cfg,
        decoder_config=pretrained_config,
        pad_index=tokenizer.pad_token_id
    )

    # -------------- Step-3: Replace the random weights with pretrained weights --------------
    # Replace the text decoder weights
    match_result = model.decoder_model.load_state_dict(
        pretrained_text_model.state_dict(),
        strict=False
    )

    # Replace the image encoder weights
    model.encoder_model.load_state_dict(pretrained_encoder_model.state_dict())


    # -------------- Step-4: Verify the weight transfer for decoder model --------------
    # Verify the layers not matched are the newly added layers
    newly_added_layers = ['cross_attention', 'layer_norm1_5', 'cross_attn_gate']
    all_matched = True
    for layer_name in match_result.missing_keys:
        is_new = any([new_layer in layer_name for new_layer in newly_added_layers])
        if not is_new:
            all_matched = False

    assert all_matched, "Some of the old layers' weights did not match"

    # -------------- Step-5: Freeze the weights --------------
    # Decoder
    # Now, freeze the old pretrained layers and allow only new layers to train
    for name, params in model.decoder_model.named_parameters():
        if name not in match_result.missing_keys:
            params.requires_grad = False

    # Do some sanity checks
    # 1 - Check if the newly added layers are the unfrozen layers
    unfrozen_layers = []
    for name, params in model.decoder_model.named_parameters():
        if params.requires_grad:
            unfrozen_layers.append(name)

    # 2 - Check the missing_keys layers and unfrozen layers match
    all_matched = True
    for name1, name2 in zip(match_result.missing_keys, unfrozen_layers):
        if name1 != name2:
            print(f"{name1} --- {name2}", flush=True)
            all_matched = False

    assert all_matched, "Some pretrained layers are not frozen!"

    # Encoder
    # Load and freeze the image encoder also
    for name, parameters in model.encoder_model.named_parameters():
        parameters.requires_grad = False

    # No need for any checks for encoder as the model configuration is the same

    # -------------- Step-6: Sanity check on the unfrozen layers --------------
    trainable_layers = []
    for name, params in model.named_parameters():
        if params.requires_grad:
            trainable_layers.append(name)

    print("List of trainable layers in the model:")
    for name in trainable_layers:
        print(name)

    # -------------- Print model sizes --------------
    encoder_params = 0
    for params in model.encoder_model.parameters():
        encoder_params += params.numel()
    print(f"No. of parameters in encoder: {encoder_params}")

    decoder_params = 0
    for params in model.decoder_model.parameters():
        decoder_params += params.numel()
    print(f"No. of parameters in decoder: {decoder_params}")

    trainable_params = 0
    for params in model.parameters():
        if params.requires_grad:
            trainable_params += params.numel()
    print(f"No. of trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {((trainable_params/(encoder_params + decoder_params))*100):.2f}%")

    # Create a random image and text
    batch_size = 4
    img_height = 64
    img_width = 256
    num_toks = 10

    inp_img = torch.randn((batch_size, 1, img_height, img_width))
    # Valid frame count per sample = real_width // downsample. Here all samples use
    # the full width, so every sample has img_width // downsample frames.
    input_lengths = torch.full((batch_size,), img_width // img_encoder_cfg.downsample, dtype=torch.long)
    inp_ids = torch.randint(1, 2048, (batch_size, num_toks))
    txt_pad_msk = torch.ones((batch_size, num_toks))

    print(input_lengths.shape)
    print(txt_pad_msk.shape)

    output = model(inp_img, inp_ids, input_lengths, txt_pad_msk)

    print(output)