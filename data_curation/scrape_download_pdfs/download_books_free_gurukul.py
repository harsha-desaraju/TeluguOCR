
import os
import re
import json
import requests
import warnings
from pathlib import Path
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


def save_webpage(url: str, save_path: str):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses

    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(response.text)



def download_file(info_dct: dict, storage_location: str):
    bid = info_dct["bid"]
    book_name = info_dct["title_english"]

    download_url = "https://old.freegurukul.org/downloadpdf1.php"

    params = {
        "bid": bid,
        "download_file": "download"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": f"https://www.freegurukul.org/view-book/{bid}",
    }

    response = requests.get(download_url, params=params, headers=headers, stream=True)
    response.raise_for_status()

    # Optional sanity check
    # print("Content-Type:", response.headers.get("Content-Type"))

    filename = f"{book_name}_{bid}.pdf"

    with open(f"{storage_location}/{filename}", "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
    print("Downloaded:", filename)



def get_books_info(url: str):
    """ Get the info about the books that can be downloaded """
    response = requests.get(url)

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all <script> tags
    scripts = soup.find_all("script")

    data_value = None

    # Loop through scripts to find the one containing "var data"
    for script in scripts[22:]:
        if script.text and "var data" in script.text:
            # Use regex to extract the JSON part
            match = re.search(r"var data\s*=\s*(\[.*?]);", script.string, re.DOTALL)
            if match:
                # Convert the string to a Python dictionary
                data_value = json.loads(match.group(1))
                break

    return data_value


def get_page_links_from_home_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tags = soup.find_all('div', class_="col-sm-6 col-md-3 col-lg-3 telugu")

    base_url = "https://old.freegurukul.org"

    links = []
    for tag in tags:
        link = f"{base_url}{tag.a.get("href")}#home"
        links.append(link)

    return links




def download_books_from_free_gurukul(output_dir: str, book_memory_limit: int = 30):
    memory_size_limit = 30
    home_page = "https://old.freegurukul.org/category#home"

    page_links = get_page_links_from_home_page(home_page)

    for link in page_links:
        books_info = get_books_info(link)
        for dct in books_info:
            file_size = int(dct['size'][:-2])
            if file_size < memory_size_limit:
                download_file(dct, storage_location=output_dir)
            else:
                print(f"Skipped!!: {dct['title_english']}...")


if __name__ == "__main__":

    output_fol = Path(__file__).parents[2] / "data/pdf_files/free_gurukul"
    os.makedirs(output_fol, exist_ok=True)

    download_books_from_free_gurukul(str(output_fol))