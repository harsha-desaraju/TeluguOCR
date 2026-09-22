
import tesserocr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def get_textline_boxes(image: Image.Image, lang="tel", plot_image: bool = False):

    with tesserocr.PyTessBaseAPI(
            path="/opt/homebrew/opt/tesseract/share/tessdata",
            lang=lang) as api:
        api.SetPageSegMode(tesserocr.PSM.AUTO)
        api.SetImage(image)
        api.Recognize()

        iterator = api.GetIterator()
        boxes = []

        while iterator:
            bbox = iterator.BoundingBox(tesserocr.RIL.TEXTLINE)
            if bbox:
                boxes.append(bbox)

            if not iterator.Next(tesserocr.RIL.TEXTLINE):
                break

    if plot_image:
        plot_img = image.copy()
        draw = ImageDraw.Draw(plot_img)

        for bbox in boxes:
            draw.rectangle(bbox)

        plt.imshow(plot_img)
        plt.show()

    return boxes


def crop_image(image: Image.Image, bboxes: list[tuple[float|int, float|int, float|int, float|int]]):
    """Crop image based on the bounding boxes provided"""

    cropped_images = []

    for bbox in bboxes:
        cimg = image.crop(bbox)
        cropped_images.append(cimg)

    return cropped_images




if __name__ == '__main__':

    from PIL import ImageDraw


    # img = Image.open('/Users/xai/Desktop/Screenshot 2026-09-21 at 2.09.44 PM.png')
    img = Image.open('/Users/xai/Desktop/page.png')
    bboxes = get_textline_boxes(img)

    print(bboxes)

    draw = ImageDraw.Draw(img)

    for bbox in bboxes:
        draw.rectangle(bbox, outline='green')

    plt.imshow(img)
    plt.show()


