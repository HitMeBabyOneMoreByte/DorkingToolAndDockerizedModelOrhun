# FileTypeDetectorBrowser.py

import os
import ssl
import random
import logging
import time
from typing import Optional, List
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager, ChromeType
from abc import ABC, abstractmethod
from selenium.webdriver.common.by import By
from urllib.parse import urlparse
import tldextract
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import tempfile

from dorking_tool.logger_setup import setup_logger

# from selenium_authenticated_proxy import SeleniumAuthenticatedProxy

# Merkezi logger'ı yapılandır
logger = setup_logger()

# stealth modülü henüz kullanılmıyor
stealth = None 

BASE_PORT = 9230

class BrowserDriver(ABC):
    @abstractmethod
    def get_driver(self):
        pass

class StandardChromeDriver(BrowserDriver):
    """
    Standart Chrome WebDriver'ı başlatmak için kullanılan sınıf.
    """

    def __init__(
        self,
        site: str,
        keywords: list,
        login_pages: list,
        instance_id: int, 
        proxies: list = None,
        user_agents: list = None,
        headless: bool = True,
        chrome_version: Optional[str] = "130.0.6723.91"
    ):
        """
        StandardChromeDriver sınıfının yapıcı metodu.
        
        Args:
            site (str): İncelenecek site URL'si.
            keywords (list): Anahtar kelimeler listesi.
            login_pages (list): Giriş sayfaları listesi.
            instance_id (int): Örneğin ID'si.
            proxies (list, optional): Kullanılacak proxy listesi.
            user_agents (list, optional): Kullanılacak kullanıcı ajanları listesi.
            headless (bool, optional): Tarayıcının headless modda çalışıp çalışmayacağı.
            chrome_version (str, optional): Kullanılacak Chrome sürümü.
        """
        self.site = site 
        self.instance_id = instance_id
        self.site_domain = ""  
        self.valid_element_urls = []
        self.depth = 0
        self.proxies = proxies if proxies else []
        self.login_pages = login_pages if login_pages else []
        self.user_agents = user_agents if user_agents else [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.91 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.91 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 7.0; HTC 10 Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.83 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; U; Android-4.0.3; en-us; Galaxy Nexus Build/IML74K) AppleWebKit/535.7 (KHTML, like Gecko) CrMo/16.0.912.75 Mobile Safari/535.7",
            "Mozilla/5.0 (Linux; Android 6.0.1; SAMSUNG SM-N910F Build/MMB29M) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/44.0.2403.133 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 5.0; SAMSUNG SM-N900 Build/LRX21V) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/2.1 Chrome/34.0.1847.76 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 6.0.1; SAMSUNG SM-G570Y Build/MMB29K) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/4.0 Chrome/44.0.2403.133 Mobile Safari/537.36",
            "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1",
            "Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4",
            "Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko",
            "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0; MDDCJS)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0; Trident/5.0)",
            "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)",
            "Mozilla/5.0 (Windows; U; MSIE 7.0; Windows NT 6.0; en-US)",
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36 Edge/14.14393",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:94.0) Gecko/20100101 Firefox/94.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; en-US) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.37 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.92 Safari/537.36",
            "Mozilla/5.0 (Android 11; Mobile; rv:89.0) Gecko/89.0 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Opera/79.0.4143.66 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        ]
        self.headless = headless
        self.chrome_version = chrome_version
        self.logger = logger  # Merkezi logger kullanılıyor
        self.setup_driver()

    def setup_driver(self):
        """
        WebDriver'ı kurar ve başlatır.
        """
        try:
            port = BASE_PORT + self.instance_id  
            user_data_dir = tempfile.mkdtemp()
            options = Options()
            seleniumwire_options = {}
            
            if self.headless:
                options.add_argument("--headless")
            options.add_argument(f'--user-data-dir={user_data_dir}')
            options.add_argument('--disable-gpu')  
            options.add_argument('--disable-software-rasterizer') 
            options.add_argument('--window-size=800,600')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-background-timer-throttling')
            options.add_argument('--disable-backgrounding-occluded-windows')
            options.add_argument('--disable-breakpad')
            options.add_argument('--disable-client-side-phishing-detection')
            options.add_argument('--disable-component-update')
            options.add_argument('--disable-hang-monitor')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-popup-blocking')
            options.add_argument('--disable-prompt-on-repost')
            options.add_argument('--disable-renderer-backgrounding')
            options.add_argument('--disable-sync')
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--allow-insecure-localhost')
            options.add_argument(f"--remote-debugging-port={port}")
            options.add_argument(f"--crash-dumps-dir={os.path.expanduser('~/tmp/Crashpad')}")

            prefs = {
                "profile.managed_default_content_settings.images": 2,  # Resimleri engelle
                "profile.managed_default_content_settings.stylesheets": 1,  
                "profile.managed_default_content_settings.javascript": 1,  
                "profile.managed_default_content_settings.plugins": 1,
                "profile.managed_default_content_settings.popups": 1,
                "profile.managed_default_content_settings.geolocation": 1,
                "profile.managed_default_content_settings.notifications": 1,
                "profile.managed_default_content_settings.media_stream": 1,
            }
            options.add_experimental_option("prefs", prefs)

            if self.user_agents:
                user_agent = random.choice(self.user_agents)
                options.add_argument(f"user-agent={user_agent}")

            if self.proxies:
                proxy = random.choice(self.proxies)
                
                # options.add_argument(f'--proxy-server={proxy}')
                
                # https://stackoverflow.com/a/77147905
                # proxy_helper = SeleniumAuthenticatedProxy(proxy_url=proxy)
                # proxy_helper.enrich_chrome_options(options)
                
                # https://stackoverflow.com/a/67703635
                seleniumwire_options['proxy'] = {
                    'http': f'http://{proxy}',
                    'https': f'https://{proxy}'
                }
                
                # 'http': 'http://user:pass@192.168.10.100:8888'

            service = Service(ChromeDriverManager(
                chrome_type=ChromeType.CHROMIUM,
                driver_version=self.chrome_version
            ).install())
            
            service.startup_timeout = 30 
            
            # self.driver = webdriver.Chrome(service=service, options=options)
            self.driver = webdriver.Chrome(service=service, options=options, seleniumwire_options=seleniumwire_options)
            
            self.logger.info("ChromeDriver başarıyla yüklendi ve başlatıldı.")

            if self.driver:
                # WebDriver'ın tespit edilmesini engelle
                self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        })
                    """
                })

                if stealth:
                    stealth(
                        driver=self.driver,
                        languages=["en-US", "en"],
                        vendor="Google Inc.",
                        platform="Win32",
                        webgl_vendor="Intel Inc.",
                        renderer="Intel Iris OpenGL Engine",
                        fix_hairline=True,
                    )
                else:
                    self.logger.warning("selenium-stealth modülü yüklü değil. Devam ediliyor...")

        except Exception as e:
            self.logger.error(f"ChromeDriver başlatılırken hata oluştu: {e}")
            self.driver = None

    def get_driver(self):
        """
        WebDriver örneğini döner.
        
        Returns:
            webdriver.Chrome: WebDriver örneği.
        """
        return self.driver

    def quit_driver(self):
        """
        WebDriver'ı düzgün bir şekilde kapatır.
        """
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver başarıyla kapatıldı.")
            except Exception as e:
                self.logger.error(f"WebDriver kapatılırken hata oluştu: {e}")
            finally:
                self.driver = None