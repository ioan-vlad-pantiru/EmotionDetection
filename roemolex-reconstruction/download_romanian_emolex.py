"""
Script helper pentru descărcarea NRC EmoLex românesc.
"""
import sys
from pathlib import Path
import requests
import zipfile
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, NRC_EMOLEX_RO_RAW, LOG_FILE
from utils import setup_logging

logger = setup_logging(LOG_FILE)


def download_nrc_romanian():
    """
    Încearcă să descarce NRC EmoLex românesc.
    """
    logger.info("=" * 60)
    logger.info("Descărcare NRC EmoLex românesc")
    logger.info("=" * 60)
    
    # URL-uri posibile
    urls = [
        "http://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon-v0.92.zip",
        "https://github.com/saifmohammad/NRC-Emotion-Lexicon/archive/refs/heads/master.zip",
    ]
    
    output_path = NRC_EMOLEX_RO_RAW
    
    # Verifică dacă există deja
    if output_path.exists():
        logger.info(f"Fișierul există deja: {output_path}")
        response = input("Vreți să-l suprascrieți? (da/nu): ")
        if response.lower() != 'da':
            return False
        output_path.unlink()
    
    # Încearcă să descarce
    for url in urls:
        try:
            logger.info(f"Încercare descărcare de la: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Descarcă într-un fișier temporar
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_path = Path(tmp_file.name)
                total_size = int(response.headers.get('content-length', 0))
                
                logger.info(f"Descărcare în curs... ({total_size / 1024 / 1024:.2f} MB)")
                
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log la fiecare MB
                                logger.info(f"  Progres: {percent:.1f}%")
                
                logger.info("Descărcare completă. Extragere...")
                
                # Extrage arhiva
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    # Caută fișierul românesc
                    romanian_files = [
                        name for name in zip_ref.namelist() 
                        if 'romanian' in name.lower() or 'ro.txt' in name.lower() or 'Romanian.txt' in name
                    ]
                    
                    if not romanian_files:
                        logger.warning("Nu s-a găsit fișierul românesc în arhivă.")
                        logger.info("Fișiere disponibile:")
                        for name in zip_ref.namelist()[:20]:
                            logger.info(f"  - {name}")
                        tmp_path.unlink()
                        continue
                    
                    # Extrage primul fișier românesc găsit
                    romanian_file = romanian_files[0]
                    logger.info(f"Găsit fișier românesc: {romanian_file}")
                    
                    # Extrage conținutul
                    content = zip_ref.read(romanian_file)
                    
                    # Scrie fișierul
                    with open(output_path, 'wb') as f:
                        f.write(content)
                    
                    logger.info(f"✓ Fișier salvat: {output_path}")
                    tmp_path.unlink()
                    return True
                    
        except Exception as e:
            logger.warning(f"Eroare la descărcare de la {url}: {e}")
            continue
    
    logger.error("Nu s-a putut descărca NRC EmoLex românesc automat.")
    logger.info("\n" + "=" * 60)
    logger.info("INSTRUCȚIUNI MANUALE:")
    logger.info("=" * 60)
    logger.info("1. Accesați: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
    logger.info("2. Descărcați versiunea 'OneFilePerLanguage'")
    logger.info("3. Extrageți arhiva și găsiți fișierul 'Romanian.txt'")
    logger.info(f"4. Copiați fișierul la: {output_path}")
    logger.info("\nSau:")
    logger.info("5. Dacă aveți deja RoEmoLex original (CSV/TSV):")
    logger.info(f"   Plasați-l la: {RAW_DIR / 'roemolex.csv'}")
    
    return False


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        logger.error("Lipsește pachetul 'requests'. Instalați cu: pip install requests")
        sys.exit(1)
    
    success = download_nrc_romanian()
    if success:
        logger.info("\n✓ Descărcare reușită!")
        logger.info("Acum puteți rula: python step1_acquire.py --no-resume")
    else:
        logger.info("\n✗ Descărcare automată eșuată. Urmați instrucțiunile de mai sus.")

