from PIL import Image
from bs4 import BeautifulSoup


def rip_out_html(text: str):
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ").replace('</p>', '</p> ')
    return BeautifulSoup(text, features="html.parser").get_text()


# copy and paste from huggingface's jupyter notebook (MIT License? S/O also contain this)
def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color=(0, 0, 0))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

