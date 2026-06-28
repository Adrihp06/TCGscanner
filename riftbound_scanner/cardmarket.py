from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from random import SystemRandom
from typing import Any

from .models import Card, Price, PriceMode

LANGUAGE_IDS = {
    "EN": "1",
    "FR": "2",
    "DE": "3",
    "ES": "4",
    "IT": "5",
    "ZH-CN": "6",
    "JA": "7",
    "PT": "8",
    "RU": "9",
    "KO": "10",
    "ZH-TW": "11",
}

COUNTRY_IDS = {
    "AT": "1",
    "BE": "2",
    "BG": "3",
    "CH": "4",
    "CY": "5",
    "CZ": "6",
    "DE": "7",
    "DK": "8",
    "EE": "9",
    "ES": "10",
    "FI": "11",
    "FR": "12",
    "GB": "13",
    "GR": "14",
    "HU": "15",
    "IE": "16",
    "IT": "17",
    "NL": "23",
    "NO": "24",
    "PL": "25",
    "PT": "26",
    "SE": "28",
}


@dataclass(frozen=True)
class OAuthCredentials:
    app_token: str
    app_secret: str
    access_token: str
    access_secret: str


class CardmarketClient:
    base_url = "https://apiv2.cardmarket.com/ws/v2.0/output.json"

    def __init__(self, credentials: OAuthCredentials) -> None:
        self.credentials = credentials
        self.random = SystemRandom()

    def get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        params = params or {}
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(url, headers={"Authorization": self._auth_header("GET", url)})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def _auth_header(self, method: str, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        oauth_params = {
            "oauth_consumer_key": self.credentials.app_token,
            "oauth_token": self.credentials.access_token,
            "oauth_nonce": str(self.random.getrandbits(64)),
            "oauth_timestamp": str(int(time.time())),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_version": "1.0",
        }
        signature_params = {**query_params, **oauth_params}
        encoded_params = urllib.parse.urlencode(sorted(signature_params.items()), quote_via=urllib.parse.quote)
        normalized_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        base = "&".join(
            urllib.parse.quote(part, safe="")
            for part in [method.upper(), normalized_url, encoded_params]
        )
        key = "&".join(
            urllib.parse.quote(part, safe="")
            for part in [self.credentials.app_secret, self.credentials.access_secret]
        )
        digest = hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()
        oauth_params["oauth_signature"] = base64.b64encode(digest).decode()
        header = ", ".join(
            f'{urllib.parse.quote(key)}="{urllib.parse.quote(value)}"'
            for key, value in oauth_params.items()
        )
        return f"OAuth {header}"


class CardmarketPriceProvider:
    def __init__(self, client: CardmarketClient) -> None:
        self.client = client

    @classmethod
    def from_env(cls) -> "CardmarketPriceProvider":
        missing = [
            name
            for name in [
                "CARDMARKET_APP_TOKEN",
                "CARDMARKET_APP_SECRET",
                "CARDMARKET_ACCESS_TOKEN",
                "CARDMARKET_ACCESS_SECRET",
            ]
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(f"Missing Cardmarket credentials: {', '.join(missing)}")
        credentials = OAuthCredentials(
            app_token=os.environ["CARDMARKET_APP_TOKEN"],
            app_secret=os.environ["CARDMARKET_APP_SECRET"],
            access_token=os.environ["CARDMARKET_ACCESS_TOKEN"],
            access_secret=os.environ["CARDMARKET_ACCESS_SECRET"],
        )
        return cls(CardmarketClient(credentials))

    def get_price(self, card: Card, language: str, seller_country: str, mode: PriceMode) -> Price:
        if card.cardmarket_product_id is None:
            return Price(
                mode=mode,
                amount=None,
                source="cardmarket",
                filters={"language": language, "seller_country": seller_country},
                message="Card is not mapped to a Cardmarket product.",
            )
        if mode == "trend":
            return self._trend_price(card, language, seller_country)
        return self._min_price(card, language, seller_country)

    def _trend_price(self, card: Card, language: str, seller_country: str) -> Price:
        payload = self.client.get(f"/products/{card.cardmarket_product_id}")
        product = payload.get("product", payload)
        guide = product.get("priceGuide") or {}
        amount = guide.get("TREND") or guide.get("trend")
        return Price(
            mode="trend",
            amount=float(amount) if amount is not None else None,
            source="cardmarket",
            filters={"language": language, "seller_country": seller_country},
            message=None if amount is not None else "Cardmarket trend price unavailable.",
        )

    def _min_price(self, card: Card, language: str, seller_country: str) -> Price:
        params = {
            "maxResults": "100",
            "start": "0",
            "minCondition": "NM",
            "isSigned": "false",
            "isAltered": "false",
        }
        language_id = LANGUAGE_IDS.get(language.upper())
        country_id = COUNTRY_IDS.get(seller_country.upper())
        if language_id:
            params["idLanguage"] = language_id
        if country_id:
            params["sellerCountry"] = country_id

        payload = self.client.get(f"/articles/{card.cardmarket_product_id}", params)
        articles = payload.get("article") or payload.get("articles") or []
        if isinstance(articles, dict):
            articles = [articles]
        prices = [
            float(article["price"])
            for article in articles
            if article.get("price") is not None
        ]
        return Price(
            mode="min",
            amount=min(prices) if prices else None,
            source="cardmarket",
            filters={"language": language, "seller_country": seller_country},
            message=None if prices else "No matching Cardmarket articles found.",
        )
