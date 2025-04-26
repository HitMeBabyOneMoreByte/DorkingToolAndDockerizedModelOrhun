# FileTypeDetector.py

import os
import random
import logging
import time
import json
import argparse
import re
import signal
import sys
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

from seleniumwire import webdriver

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests  
from functools import wraps
import tldextract

# Yüklemeleri ve yardımcı modülleri içe aktar
load_dotenv()

from detectors.util import get_main_domain
from dorking_tool.logger_setup import setup_logger

# Merkezi logger'ı yapılandır
logger = setup_logger()

# Desteklenen dosya türleri
SUPPORTED_TYPES = {
    "DOC_or_DOCX": [".doc", ".docx"],
    "XLS_or_XLSX": [".xls", ".xlsx"],
    "Web_Page": [".html", ".htm", ""],
    "PDF": [".pdf"],
    "JSON": [".json"],
    "Text_File": [".txt"],
    "CSV": [".csv"],
    "UTF8": ["utf8"],
}

# Dosya uzantısı ile tür eşlemesi
EXTENSION_MAPPING = {
    '.pdf': 'PDF',
    '.doc': 'DOC_DOCX',
    '.docx': 'DOC_DOCX',
    '.xls': 'XLS_XLSX',
    '.xlsx': 'XLS_XLSX',
    '.html': 'Web_Page',
    '.htm': 'Web_Page',
    '.txt': 'Text_File',
    '.csv': 'CSV',
    '.json': 'JSON',
    '.png': 'PNG',
    '.jpg': 'JPG',
    '.jpeg': 'JPG',
    '.gif': 'GIF',
    '.bmp': 'BMP',
}

# MIME tipi ile tür eşlemesi
MIME_MAPPING = {
    'application/pdf': 'PDF',
    'application/vnd.ms-excel': 'XLS_XLSX',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLS_XLSX',
    'application/msword': 'DOC_DOCX',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOC_DOCX',
    'text/html': 'Web_Page',
    'text/plain': 'Text_File',
    'application/json': 'JSON',
    'text/csv': 'CSV',
    'image/png': 'PNG',
    'image/jpeg': 'JPG',
    'image/gif': 'GIF',
    'image/bmp': 'BMP',
    'application/octet-stream': 'Unsupported_Extensions',
}

def signal_handler(sig, frame):
    """
    Programın düzgün bir şekilde kapanmasını sağlar.
    """
    logger.info("SIGINT alındı. Program kapatılıyor...")
    print("\nProgram sonlandırılıyor. İşlemler güvenli bir şekilde sonlandırıldı.")
    sys.exit(0)

# SIGINT sinyalini yakala
signal.signal(signal.SIGINT, signal_handler)

def retry(ExceptionToCheck, tries=3, delay=2, backoff=2):
    """
    Belirtilen istisnalar için fonksiyonu yeniden denemek üzere bir dekoratör.
    
    Args:
        ExceptionToCheck (Exception): Yeniden denemek istenen istisna türü.
        tries (int): Deneme sayısı.
        delay (int): İlk denemeden sonra beklenen süre (saniye).
        backoff (int): Bekleme süresinin artış katsayısı.
    
    Returns:
        function: Dekore edilmiş fonksiyon.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = f"{e}, {mdelay} saniye içinde tekrar denenecek..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

def get_file_extension(url):
    """
    URL'den dosya uzantısını çıkarır.
    
    Args:
        url (str): İncelenecek URL.
    
    Returns:
        str: Dosya uzantısı (örn. '.pdf') veya boş string.
    """
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        extension = os.path.splitext(path)[1].lower()
        extension = unquote(extension) 
        logger.debug(f"URL: {url} - Çıkarılan dosya uzantısı: {extension}")
        return extension
    except Exception as e:
        logger.error(f"URL'den dosya uzantısı çıkarılırken hata oluştu: {url}. Hata: {e}")
        return ""

def extract_filename_from_content_disposition(content_disposition):
    """
    Content-Disposition başlığından dosya adını çıkarır.
    
    Args:
        content_disposition (str): Content-Disposition başlığı.
    
    Returns:
        str or None: Dosya adı veya None.
    """
    if not content_disposition:
        return None
    fname = re.findall(r'filename\=.\'\'(.+)', content_disposition, re.IGNORECASE)
    if not fname:
        fname = re.findall(r'filename="?(.+)"?', content_disposition, re.IGNORECASE)
    return fname[0] if fname else None

@retry(Exception, tries=3, delay=2, backoff=2)
def get_file_type_requests(url, timeout=10):
    """
    HTTP HEAD isteği kullanarak dosya türünü belirler.
    
    Args:
        url (str): İncelenecek URL.
        timeout (int): Zaman aşımı süresi (saniye).
    
    Returns:
        str: Belirlenen dosya türü.
    """
    try:
        logger.debug(f"URL: {url} - HEAD isteği yapılıyor.")
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        content_type = response.headers.get('Content-Type', '').split(';')[0].lower()
        content_disposition = response.headers.get('Content-Disposition', '')
        logger.debug(f"URL: {url} - Content-Type: {content_type}, Content-Disposition: {content_disposition}")

        type_name = MIME_MAPPING.get(content_type, "Unsupported_or_unsupported_file_type")
        
        if type_name == "Unsupported_or_unsupported_file_type" and content_disposition:
            filename = extract_filename_from_content_disposition(content_disposition)
            if filename:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in EXTENSION_MAPPING:
                    type_name = EXTENSION_MAPPING[file_ext]
                    logger.info(f"URL: {url} - Dosya adı üzerinden belirlenen tür: {type_name} (Filename: {filename})")
        if type_name != "Unsupported_or_unsupported_file_type":
            logger.info(f"URL: {url} - HTTP başlıkları üzerinden belirlenen dosya türü: {type_name}")
            return type_name
        else:
            raise ValueError("Content-Type belirlenemedi.")
    except requests.RequestException as e:
        logger.error(f"URL: {url} - HTTP isteği sırasında hata: {e}")
        raise e

@retry(Exception, tries=3, delay=2, backoff=2)
def get_file_type_selenium(url, driver, timeout=30):
    """
    Selenium WebDriver kullanarak dosya türünü belirler.
    
    Args:
        url (str): İncelenecek URL.
        driver (webdriver): Selenium WebDriver örneği.
        timeout (int): Zaman aşımı süresi (saniye).
    
    Returns:
        str: Belirlenen dosya türü.
    """
    parsed_url = urlparse(url)
    query_params = parsed_url.query.lower()

    if 'format=utf8' in query_params:
        type_name = 'Unsupported_or_unsupported_file_type'
        logger.info(f"URL: {url} - 'format=utf8' parametresi bulundu. Tür: {type_name}")
        return type_name

    try:
        logger.debug(f"URL: {url} - Selenium WebDriver ile yükleniyor.")
        driver.get(url)
    except TimeoutException:
        logger.warning(f"URL: {url} - Selenium yüklenirken zaman aşımına uğradı.")
        raise TimeoutException("Selenium yüklenirken zaman aşımı.")
    except Exception as e:
        logger.error(f"URL: {url} - Selenium ile yüklenirken hata: {e}")
        raise e

    time.sleep(2)  # Sayfanın yüklenmesi için kısa bir bekleme

    for request in driver.requests:
        if request.response and request.url == url:
            content_type = request.response.headers.get('Content-Type', '').split(';')[0].lower()
            content_disposition = request.response.headers.get('Content-Disposition', '')
            logger.debug(f"URL: {url} - Content-Type: {content_type}, Content-Disposition: {content_disposition}")

            type_name = MIME_MAPPING.get(content_type, "Unsupported_or_unsupported_file_type")
            
            if type_name == "Unsupported_or_unsupported_file_type" and content_disposition:
                filename = extract_filename_from_content_disposition(content_disposition)
                if filename:
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in EXTENSION_MAPPING:
                        type_name = EXTENSION_MAPPING[file_ext]
                        logger.info(f"URL: {url} - Content-Disposition üzerinden belirlenen tür: {type_name} (Filename: {filename})")

            if type_name != "Unsupported_or_unsupported_file_type":
                logger.info(f"URL: {url} - HTTP başlıkları üzerinden belirlenen dosya türü: {type_name}")
                return type_name

    logger.info(f"URL: {url} - Selenium WebDriver kullanılarak dosya türü belirlenemedi. Tür: 'Unsupported_or_unsupported_file_type'")
    return "Unsupported_or_unsupported_file_type"

class FileTypeDetector:
    """
    URL'nin dosya türünü belirlemek için kullanılan sınıf.
    Bilinen uzantı varsa doğrudan belirler, yoksa HTTP HEAD isteği ve Selenium WebDriver kullanır.
    """

    def __init__(self, proxy_list=None, user_agents=None, headless=True):
        """
        FileTypeDetector sınıfının yapıcı metodu.
        
        Args:
            proxy_list (list, optional): Kullanılacak proxy listesi.
            user_agents (list, optional): Kullanılacak kullanıcı ajanları.
            headless (bool, optional): Tarayıcının headless modda çalışıp çalışmayacağı.
        """
        self.proxy_list = proxy_list
        self.user_agents = user_agents
        self.headless = headless

    def determine_file_type(self, url):
        """
        URL'nin dosya türünü belirler.
        
        Args:
            url (str): İncelenecek URL.
        
        Returns:
            str: Belirlenen dosya türü.
        """
        # Öncelikle dosya uzantısını kontrol et
        extension = get_file_extension(url)
        if extension in EXTENSION_MAPPING:
            type_name = EXTENSION_MAPPING[extension]
            logger.info(f"URL: {url} - Dosya uzantısı {extension} üzerinden belirlenen tür: {type_name} (WebDriver gerekmez)")
            return type_name

        # Uzantı bilinmiyorsa HTTP HEAD isteği ile dene
        try:
            type_name = get_file_type_requests(url)
            return type_name
        except Exception as e:
            logger.warning(f"URL: {url} - HTTP başlıkları kullanılarak dosya türü belirlenemedi: {e}")

        # Hala belirlenemediyse Selenium WebDriver kullan
        from detectors.FileTypeDetectorBrowser import StandardChromeDriver  # Gerekli modülü içe aktar
        proxy = random.choice(self.proxy_list) if self.proxy_list else None
        browser_driver = StandardChromeDriver(
            site=url,
            keywords=[],
            login_pages=[],
            instance_id=1,
            proxies=[proxy] if proxy else None,
            user_agents=self.user_agents,
            headless=self.headless
        )
        driver = browser_driver.get_driver()

        try:
            type_name = get_file_type_selenium(url, driver)
        except Exception as e:
            logger.error(f"URL: {url} - Selenium kullanılarak dosya türü belirlenemedi: {e}")
            type_name = "Unsupported_or_unsupported_file_type"
        finally:
            browser_driver.quit_driver()

        return type_name