
import fitz
from pathlib import Path
from datasets import Dataset, Features, Value, Image as DImage
from PIL import Image
import io
from joblib import Parallel, delayed


def extract_lines_with_images(file_path, dpi=300, padding=0):
    """
    Extract lines from a PDF with:
    - line text
    - bounding box
    - page number
    - PIL image of the line crop

    Args:
        file_path (str): Path to PDF
        dpi (int): Render DPI
        padding (int): Padding around bbox

    Returns:
        List[dict]
    """
    try:
        doc = fitz.open(file_path)
        results = []
        for page_number, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            for block in data["blocks"]:
                # Skip non-text blocks
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text_parts = []
                    for span in line.get("spans", []):
                        line_text_parts.append(span.get("text", ""))
                    line_text = "".join(line_text_parts).strip()
                    if not line_text:
                        continue
                    bbox = line["bbox"]
                    rect = fitz.Rect(bbox)

                    # Add padding to avoid clipping Telugu glyphs
                    rect.x0 -= padding
                    rect.y0 -= padding
                    rect.x1 += padding
                    rect.y1 += padding

                    # Render cropped region
                    pix = page.get_pixmap(
                        clip=rect,
                        dpi=dpi
                    )

                    # Convert Pixmap -> PIL Image
                    img_bytes = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_bytes))

                    results.append({
                        "text": line_text,
                        "bbox": bbox,
                        "page": page_number,
                        "image": pil_image
                    })
        doc.close()
    except Exception as e:
        results = []
    return results



if __name__ == '__main__':

    MIN_IMAGE_WIDTH = 50

    ds_features = Features({
        "image": DImage(),
        "text": Value("string"),
        "image_width": Value("int64"),
        "file_name": Value("string"),
        "page_number": Value("int64")
    })

    files = list(Path("/Users/xai/Personal/Projects/TeluguOCR/data/pdf_files/digital").rglob("*.pdf"))
    valid_books = [
        "Panchatantram_Vishnu Sharma.pdf",
        "Bharata Savitri_Veda Vyasa.pdf",
        "Ramayanamu_Atukuri Molla.pdf",
        "Bhagavad Gita_Vyasa.pdf",
        "Sampurna Neetichandrika_Bulusu Sitaramasastry.pdf",
        "Vishnu Stotra Mala_Various.pdf",
        "Asamardhuni Jeevayatra_Tripuraneni Gopichand.pdf",
        "Krutulu_Adi Shankaracharya.pdf",
        "Satakamulu_Various.pdf",
        "Bala Vyakaranamu_Paravastu Chinnayasuri.pdf",
        "Neela Sundari Parinayam_Koochimanchi Timmana.pdf",
        "Hara Vilasamu_Srinatha.pdf",
        "Bhetala Kathalu_Burela Stayanarayanmurty.pdf",
        "Devi Stotra Ratnavali_Various.pdf",
        "Devulapalli Krishnasastri Krutulu.pdf",
        "Vara Vikrayam_Kallakuri Narayanarao.pdf",
        "Shiva Stotra Mala_Various.pdf",
        "Sri Gita Govindam_Jayadeva.pdf",
        "Kanyasulkam_Gurajada Apparao.pdf",
        "Mithunam - Sri Ramana.pdf",
        "Rajasekhara Charitramu_Kandukuri Veeresalingam.pdf",
        "Janardanashtakam_Kandukuri Rudrakavi.pdf",
        "Ganapati_Chilakamarthi Lakshminarasimham.pdf",
        "Kinnerasani Patalu_Viswanatha Satyanarayana.pdf",
        "Maha Prasthanam - Srirangam Srinivasarao.pdf",
        "శ్రీ వినాయక వ్రతకల్పము.pdf",
        "Khadga Srushti - Srirangam Srinivasarao.pdf",
    ]

    files = [file for file in files if file.name in valid_books]

    with Parallel(n_jobs=8) as parallel:
        files_line_images = parallel([delayed(extract_lines_with_images)(str(file)) for file in files])

    ds_buffer = []
    for file, line_images in zip(files, files_line_images):
        print(file.name)
        for line_img in line_images:
            img = line_img["image"]
            img_width = img.size[0]
            if img_width >= MIN_IMAGE_WIDTH:
                ds_buffer.append({
                    "image": line_img["image"],
                    "text": line_img["text"],
                    "image_width": img_width,
                    "file_name": file.name,
                    "page_number": line_img["page"]
                })

    ds = Dataset.from_list(ds_buffer, ds_features)

    ds.push_to_hub(
        "harsha-desaraju/telugu-line-text-image",
        commit_message=f"upload text and images",
    )