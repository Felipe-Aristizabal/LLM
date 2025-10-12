# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .config import PipelineConfig


def _build_driver(cfg: PipelineConfig) -> webdriver.Chrome:
    options = Options()
    if cfg.headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={cfg.window_size}")
    options.add_argument(f"--user-agent={cfg.user_agent}")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()), options=options
    )


def _auto_scroll(driver: webdriver.Chrome, pause: float, steps: int) -> None:
    """Scroll incremental para disparar carga perezosa (lazy load)."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(steps):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def _try_accept_cookies(driver: webdriver.Chrome, timeout: int = 4) -> bool:
    """
    Intenta aceptar el banner de cookies si aparece.
    Devuelve True si hizo clic en algún botón, False en caso contrario.
    """
    clicked = False

    xpath_candidates = [
        "//button[contains(translate(., 'ACEPTAR', 'aceptar'), 'aceptar')]",
        "//button[contains(., 'Aceptar')]",
        "//button[contains(., 'ACEPTAR')]",
        "//button[contains(., 'Aceptar todas')]",
        "//button[contains(., 'Aceptar todo')]",
        "//button[contains(., 'OK')]",
        "//button[contains(., 'Accept')]",
        "//button[contains(., 'AGREE')]",
        "//button[contains(., 'Allow all')]",
        "//button[@id='onetrust-accept-btn-handler']",
        "//button[contains(@class, 'accept')]",
        "//button[contains(@class, 'ot-accept')]",
        "//button[contains(@id, 'sp-accept') or contains(@id, 'didomi') or contains(@id, 'consent')]",
    ]

    css_candidates = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label*='Accept']",
        "button[aria-label*='Aceptar']",
        "button[class*='accept']",
        "button[class*='ot-accept']",
        "button[class*='consent']",
        "button[class*='agree']",
    ]

    iframe_candidates = [
        "iframe[id^='sp_message_iframe_']",
        "iframe[src*='consent']",
        "iframe[src*='privacy']",
        "iframe[id*='ot-']",
        "iframe[id*='onetrust']",
        "iframe[id*='didomi']",
    ]

    def _click_elements(elems) -> bool:
        for el in elems:
            try:
                WebDriverWait(driver, min(timeout, 3)).until(
                    EC.element_to_be_clickable(el)
                )
                try:
                    el.click()
                except Exception:
                    # Si hay overlay o intercepción, intentar con JS
                    driver.execute_script("arguments[0].click();", el)
                return True
            except Exception:
                continue
        return False

    # Intento 1: buscar en DOM principal (XPaths)
    try:
        for xp in xpath_candidates:
            try:
                elems = WebDriverWait(driver, timeout).until(
                    EC.presence_of_all_elements_located((By.XPATH, xp))
                )
                if _click_elements(elems):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    # Intento 2: CSS en DOM principal
    try:
        for cs in css_candidates:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, cs)
                if elems and _click_elements(elems):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    # Intento 3: dentro de iframes de consentimiento
    try:
        iframes = []
        for sel in iframe_candidates:
            iframes.extend(driver.find_elements(By.CSS_SELECTOR, sel))

        for frame in iframes:
            try:
                driver.switch_to.frame(frame)
                # Repetir búsqueda de botones dentro del iframe
                for xp in xpath_candidates:
                    try:
                        elems = driver.find_elements(By.XPATH, xp)
                        if elems and _click_elements(elems):
                            clicked = True
                            break
                    except Exception:
                        continue
                if not clicked:
                    for cs in css_candidates:
                        try:
                            elems = driver.find_elements(By.CSS_SELECTOR, cs)
                            if elems and _click_elements(elems):
                                clicked = True
                                break
                        except Exception:
                            continue
            except Exception:
                pass
            finally:
                driver.switch_to.default_content()
            if clicked:
                break
    except Exception:
        pass

    return clicked


def fetch_html_selenium(
    url: str, cfg: PipelineConfig
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Devuelve (html, final_url). Usa scroll para maximizar contenido cargado."""
    driver = _build_driver(cfg)
    try:
        driver.get(url)

        try:
            time.sleep(max(1, cfg.selenium_wait_sec // 2))
            _try_accept_cookies(driver, timeout=4)
        except Exception:
            pass

        time.sleep(cfg.selenium_wait_sec)

        if cfg.auto_scroll:
            _auto_scroll(driver, cfg.scroll_pause, cfg.max_scroll_steps)
            try:
                _try_accept_cookies(driver, timeout=3)
            except Exception:
                pass

        html = driver.page_source

        try:
            visible_text = driver.execute_script("return document.body.innerText || ''")
        except Exception:
            visible_text = ""
        final_url = driver.current_url
        return (html, final_url, visible_text)

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Selenium falló en {url}: {exc}")
        return None, None
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def collect_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    from urllib.parse import urljoin

    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        out.append(urljoin(base_url, href))
    return out


def same_host(u1: str, u2: str) -> bool:
    return urlparse(u1).netloc.lower() == urlparse(u2).netloc.lower()
