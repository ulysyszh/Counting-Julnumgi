import os
import urllib.request


def download_font():
    """Download NanumGothic font if not exists"""
    font_path = "assets/NanumGothic.ttf"

    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)

    if not os.path.exists(font_path):
        # Updated font URL from Google Fonts
        font_url = "https://fonts.gstatic.com/s/nanumgothic/v21/PN_3Rfi-oW3hYwmKDpxS7F_z-7rJxHVIsPV5MbNO2rV2_va-6z4.ttf"
        try:
            urllib.request.urlretrieve(font_url, font_path)
            print("폰트 다운로드 완료")
        except Exception as e:
            print(f"폰트 다운로드 실패: {e}")
            print(
                "나눔고딕 폰트를 수동으로 다운로드하여 'assets/NanumGothic.ttf'에 저장해주세요."
            )
            # Create empty font file to prevent repeated download attempts
            with open(font_path, "wb") as f:
                f.write(b"")
