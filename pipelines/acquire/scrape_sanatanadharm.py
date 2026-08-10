
import os
import json
import hashlib
from pathlib import Path
from pipelines.acquire.scraper import WebScraper
from urllib.parse import unquote, urlparse





def save_file(url, content):
    decoded_url = unquote(url)

    flat_name = urlparse(decoded_url).path.strip("/").split('/')[-1].split('.')[0]

    # Truncate by bytes (Telugu chars are 3 bytes each in UTF-8)
    encoded = flat_name.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        flat_name = encoded[:MAX_BYTES].decode("utf-8", errors="ignore")

    # Append short hash to avoid collisions between truncated names
    short_hash = hashlib.md5(decoded_url.encode()).hexdigest()[:8]
    file_name = f"{flat_name}_{short_hash}.json"
    out_path = Path(OUT_DIRECTORY) / file_name

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"url": decoded_url, "content": content}, f, ensure_ascii=False)



if __name__ == '__main__':

    OUT_DIRECTORY = Path(__file__).parents[2] / "data/pdf_files/sanatanadharm"
    p_min, p_max = 3, 10
    break_min, break_max = 30, 180
    MAX_BYTES = 180

    BASE_URLs = [
        "https://sanatanadharm.com/mahabaratham/home%20mahaBratham.html",
        "https://sanatanadharm.com/links%20ramayanam/ramayanam.html",
        "https://sanatanadharm.com/%E0%B0%A6%E0%B1%87%E0%B0%B5%E0%B0%BF%E0%B0%A6%E0%B1%87%E0%B0%B5%E0%B0%A4%E0%B0%B2%E0%B1%81/18%20%E0%B0%85%E0%B0%B7%E0%B1%8D%E0%B0%9F%E0%B0%BE%E0%B0%A6%E0%B0%B6%20%E0%B0%AA%E0%B1%81%E0%B0%B0%E0%B0%BE%E0%B0%A3%E0%B0%BE%E0%B0%B2%E0%B1%81/%E0%B0%85%E0%B0%B7%E0%B1%8D%E0%B0%9F%E0%B0%BE%E0%B0%A6%E0%B0%B6%20%E0%B0%AA%E0%B1%81%E0%B0%B0%E0%B0%BE%E0%B0%A3%E0%B0%BE%E0%B0%B2%E0%B1%81.html",
        "https://sanatanadharm.com/bhagavath%20geetha/bhagavathgeetha.html"
    ]
    TARGET_PREFIX = ""

    os.makedirs(OUT_DIRECTORY, exist_ok=True)


    for BASE_URL in BASE_URLs:

        visited_links = set()

        with WebScraper(out_dir=OUT_DIRECTORY, timeout=30.0, target_prefix=TARGET_PREFIX) as scraper:
            try:
                # Scrape the website
                result = scraper.extract_page(BASE_URL)

                file_links = result.file_links

                print(len(file_links), BASE_URL)

                for link in file_links:
                    scraper.download(link)
                    print(link)

                save_file(result.url, result.text)


            except Exception as e:
                print(str(e))





