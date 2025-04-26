# detectors/util.py

import os
import requests
import io
import logging
from urllib.parse import urlparse
import tldextract
from dorking_tool.logger_setup import setup_logger

logger = setup_logger()

def get_main_domain(url):
    try:
        url = ensure_url_scheme(url)  # URL başına http/https eklenmemişse ekler.
        extracted = tldextract.extract(url)
        if extracted.registered_domain:
            return extracted.registered_domain.lower()
        else:
            parsed = urlparse(url)
            if parsed.netloc:
                return parsed.netloc.lower()
            return "Unknown"
    except Exception as e:
        logger.error(f"Error extracting main domain from URL: {url}. Error: {e}")
        return "Unknown"


def is_valid_tc_kimlik_no(tc_no):
    """
    Validates a Turkish TC Identification Number.
    """
    if not tc_no.isdigit() or len(tc_no) != 11:
        return False
    digits = list(map(int, tc_no))

    if digits[0] == 0:
        return False

    # Formula for the 10th digit
    odd_sum = digits[0] + digits[2] + digits[4] + digits[6] + digits[8]
    even_sum = digits[1] + digits[3] + digits[5] + digits[7]
    tenth_digit_calc = ((7 * odd_sum) - even_sum) % 10
    if digits[9] != tenth_digit_calc:
        return False

    # Formula for the 11th digit
    total_sum = sum(digits[:10])
    eleventh_digit_calc = total_sum % 10
    if digits[10] != eleventh_digit_calc:
        return False

    return True

def download_file(url, download_dir='downloads'):
    """
    Downloads a file from the specified URL to the given directory.

    Args:
        url (str): The URL of the file to download.
        download_dir (str): The directory where the file will be saved.

    Returns:
        str or None: The path to the downloaded file, or None if download fails.
    """
    try:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        local_filename = os.path.join(download_dir, os.path.basename(url.split('?')[0]))

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
                          '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }

        with requests.get(url, headers=headers, stream=True, timeout=30, verify=False) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"Dosya indirildi: {url} -> {local_filename}")
        return local_filename
    except requests.exceptions.RequestException as e:
        logger.error(f"Dosya indirilemedi: {url}, İstek Hatası: {e}")
    except Exception as e:
        logger.error(f"Dosya indirilemedi: {url}, Genel Hata: {e}")
    return None


def ensure_url_scheme(url, default_scheme='https'):
    parsed = urlparse(url)
    if not parsed.scheme:
        corrected_url = f"{default_scheme}://{url}"
        if is_valid_url(corrected_url):
            logger.debug(f"Added scheme to URL: {url} -> {corrected_url}")
            return corrected_url
        else:
            logger.error(f"Invalid URL after adding scheme: {corrected_url}")
            return None
    return url


def is_valid_url(url):
    """
    Validates the URL format.
    
    Args:
        url (str): The URL to validate.
    
    Returns:
        bool: True if URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False