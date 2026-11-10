import os
import pandas as pd
import time
import re
import copy
import yaml
from multiprocessing import Pool
from sys import platform
from timeit import default_timer as timer
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def prepare_selenium_on_chrome(chrome_drive_path):
    # Set up Selenium options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Replace with the path to your chromedriver
    service = Service(chrome_drive_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    return driver

def get_all_urls(base_url, driver):
    """Fetch all URLs from the base page."""
    driver.get(base_url)
    time.sleep(2)  # Allow some time for the page to load
    
    # Find all anchor tags and extract their href attributes
    links = driver.find_elements(By.TAG_NAME, "a")
    urls = set()
    for link in links:
        href = link.get_attribute("href")
        if href and href.startswith(base_url):
            urls.add(href)
    return list(urls)

def get_new_urls_within_base(url, base_url, known_urls, driver):
    """Check if a URL has new URLs within the same base URL, excluding those already known."""
    driver.get(url)
    time.sleep(2)  # Allow some time for the page to load
    links = driver.find_elements(By.TAG_NAME, "a")
    new_urls = set()
    for link in links:
        href = link.get_attribute("href")
        if href and href.startswith(base_url) and "#" not in href and href not in known_urls:
            new_urls.add(href)
    return list(new_urls)

def get_page_text(url, driver):
    """Fetch the text content of a given URL, excluding elements with the class 'sidebar'."""
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        sidebar_elements = driver.find_elements(By.CLASS_NAME, "sidebar")
        for sidebar in sidebar_elements:
            driver.execute_script("arguments[0].remove();", sidebar)

        footer_elements = driver.find_elements(By.CLASS_NAME, "footer")
        for footer in footer_elements:
            driver.execute_script("arguments[0].remove();", footer)

        table_elements = driver.find_elements(By.CLASS_NAME, "table")
        for table in table_elements:
            driver.execute_script("arguments[0].remove();", table)

        toc_elements = driver.find_elements(By.CLASS_NAME, "toc")
        for toc in toc_elements:
            driver.execute_script("arguments[0].remove();", toc)

        #return driver.find_element(By.TAG_NAME, "body").text
        return driver.page_source
    except TimeoutException:
        print(f"Timeout while loading {url}")
        return ""

def get_text_with_code_spacing(tag):
    parts = []
    for content in tag.contents:
        if isinstance(content, str):
            parts.append(content.strip())
        elif content.name in ["code", "a"]:
            parts.append(content.get_text(strip=False))
        else:
            parts.append(content.get_text(strip=True))
    return ' '.join(parts)

def extract_data_from_html_nested(url, html, driver):
    soup = BeautifulSoup(html, 'html.parser')
    results = []

    heading_stack = []
    current_descriptions = []
    current_description_buffer = []

    def get_level(tag_name):
        return int(tag_name[1])

    def flush_stack_to_level(level):
        while heading_stack and get_level(heading_stack[-1][0]) >= level:
            heading_stack.pop()
            current_descriptions.pop()

    def get_full_title():
        return " - ".join(h[1] for h in heading_stack)

    def clean_soup_section(div):
        div_copy = copy.deepcopy(div)
        for tag in div_copy.select('.code-example, .language-yaml'):
            tag.replace_with('<YAML>')
        return div_copy

    def is_valid_yaml(text):
        try:
            parsed = yaml.safe_load(text)
            return isinstance(parsed, (dict, list))
        except Exception:
            return False

    def get_full_description(extra_parts=None):
        skip_h1 = (
            len(current_descriptions[0]) > 1 and
            heading_stack and heading_stack[0][0] == 'h1'
        )
        desc_range = current_descriptions[1:] if skip_h1 else current_descriptions

        # Merge extra_parts with all description parts
        all_parts = []
        for desc in desc_range:
            all_parts.extend(desc)
        if extra_parts:
            all_parts.extend(extra_parts)

        # Remove duplicate parts (ignoring spaces), keeping the one with more characters
        deduped_dict = {}
        for part in all_parts:
            cleaned_part = re.sub(r'^\s*<YAML>\s*', '', part)
            cleaned_part = re.sub(r'^\s*YAML[\s:,-]*', '', cleaned_part, flags=re.IGNORECASE)
            cleaned_part = re.sub(r'\s*<YAML>\s*', ' ', cleaned_part)
            cleaned_part = cleaned_part.strip()
            norm = re.sub(r'\s+', '', cleaned_part)

            # Keep the part with more characters (before removing spaces)
            if norm:
                if norm not in deduped_dict or len(cleaned_part) > len(deduped_dict[norm]):
                    deduped_dict[norm] = cleaned_part
        deduped = list(deduped_dict.values())
        full_text = " ".join(deduped).strip()

        # Remove everything after the first occurrence of 'YAML' (case-insensitive)
        match = re.search(r'(and produces the following build output:)', full_text, re.IGNORECASE)
        if match:
            full_text = full_text[:match.start()]
        full_text = '.' if full_text == '' else full_text
        return full_text

    body = soup.body
    if not body:
        return []

    # Collect divs to remove first
    to_remove = []
    for div in soup.find_all('div', style=True):
        style = div.get('style', '')
        if style and 'display:none' in style.replace(' ', '').lower():
            to_remove.append(div)
    for div in to_remove:
        div.decompose()

    for element in body.find_all(recursive=False):
        for current in element.descendants:
            if not hasattr(current, 'name'):
                continue

            tag = current.name
            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                flush_stack_to_level(get_level(tag))
                heading_stack.append((tag, get_text_with_code_spacing(current), []))
                current_descriptions.append([])
                current_description_buffer = []  # Reset buffer on new heading

            elif tag == 'pre':
                code = current.find('code', class_=lambda c: c and 'language-yaml' in c)
                if code:
                    # Create entry with current buffer
                    results.append({
                        "URL": url,
                        "Title": get_full_title(),
                        "Description": get_full_description(current_description_buffer),
                        "Code": code.text
                    })
                    current_description_buffer = []  # Reset buffer after creating entry

            elif tag == 'div':
                # Check if the <div> tag has a class containing 'language-yaml'
                if ('language-yaml' in current.get('class', []) or 'highlight' in current.get('class', []) or 'code-block' in current.get('class', [])) and current.find('code'):
                    code_text = current.text
                    if is_valid_yaml(code_text):
                        results.append({
                            "URL": url,
                            "Title": get_full_title(),
                            "Description": get_full_description(current_description_buffer),
                            "Code": code_text
                        })
                    current_description_buffer = []  # Reset buffer after creating entry
            elif tag in ['p', 'div']:
                if heading_stack:
                    if 'code-example' not in current.get('class', []):
                        cleaned_current = clean_soup_section(current)
                        desc_text = get_text_with_code_spacing(cleaned_current)
                        current_description_buffer.append(desc_text)
                        current_descriptions[-1].append(desc_text)

    return results

def save_text_to_file(url, text, ci_output_dir, driver):
    filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:200] + ".csv"
    filepath = os.path.join(ci_output_dir, filename)
    data = extract_data_from_html_nested(url, text, driver)
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(filepath, index=False)

def run_crawling_for_one_base_url(chrome_drive_path, base_url, output_dir):
    start = timer()

    from selenium.webdriver.common.by import By
    selenium_driver = prepare_selenium_on_chrome(chrome_drive_path)
    os.makedirs(output_dir, exist_ok=True)
    
    title = base_url["Title"]
    base_url = base_url["URL"]
    ci_output_dir = os.path.join(output_dir, title)
    os.makedirs(ci_output_dir, exist_ok=True)
    
    print(f"Fetching URLs from {base_url} ({title})...")
    known_urls = set()
    urls = get_all_urls(base_url, selenium_driver)
    known_urls.update(urls)
    print(f"Found {len(urls)} URLs for {title}.")

    for idx, url in enumerate(urls):
        print(f"[{idx + 1}/{len(urls)}] Visiting {url}...")
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_")[:200] + ".csv"
        filepath = os.path.join(ci_output_dir, filename)
        if os.path.exists(filepath):
            print(f"    File {filename} already exists. Skipping...")
        else:
            text = get_page_text(url, selenium_driver)
            if text:
                save_text_to_file(url, text, ci_output_dir, selenium_driver)
        
        # Check for new URLs within the base URL from this page
        new_urls = get_new_urls_within_base(url, base_url, known_urls, selenium_driver)
        if new_urls:
            print(f"Found {len(new_urls)} new URLs within the base URL from {url}.")
            known_urls.update(new_urls)
            urls.extend(new_urls)

    print(f"Finished crawling {title}.")
    selenium_driver.quit()
    
    end = timer()
    print(title, "took:", (end - start)/60) # Time in minutes

def combine_csv_files(input_dir, output_file):
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Add ID column starting from 1
    combined_df.insert(0, 'ID', range(1, len(combined_df) + 1))
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    chrome_drive_path = "chromedriver" # Add the path to chromedriver
    output_dir = os.path.join(os.path.dirname(__file__), '../data')

    try:
        base_url = {"Title": "GitHubActions", "URL": "https://docs.github.com/en/actions"}
        run_crawling_for_one_base_url(chrome_drive_path, base_url, output_dir)
        combine_csv_files(f"{output_dir}/{base_url['Title']}", f"{output_dir}/{base_url['Title']}_Docs.csv")
    except Exception as ex:
        print(ex)
