import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin

# Create directories to store files
def setup_directories():
    categories = ['prelude', 'sonata', 'fugue', 'concerto', 'suite', 'aria', 'overture', 'sarabande', 'allemande']
    os.makedirs('handel_midi', exist_ok=True)
    for category in categories:
        os.makedirs(f'handel_midi/{category}', exist_ok=True)
    # For files that don't match any known category
    os.makedirs('handel_midi/other', exist_ok=True)

# Determine the category of a file based on its name and description
def get_category(title, description):
    categories = {
        'prelude': ['prelude', 'preludio'],
        'sonata': ['sonata', 'sonat'],
        'fugue': ['fugue', 'fuga'],
        'concerto': ['concerto', 'concert'],
        'suite': ['suite', 'suit'],
        'aria': ['aria', 'air'],
        'overture': ['overture', 'ouverture'],
        'sarabande': ['sarabande'],
        'allemande': ['allemande', 'allamanda']
    }
    
    title_desc = (title + ' ' + description).lower()
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in title_desc:
                return category
    
    return 'other'

# Clean filename to be valid
def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

# Download a single MIDI file
def download_midi(url, title, description):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Determine category
            category = get_category(title, description)
            
            # Create a clean filename
            filename = clean_filename(f"{title}-{description}.mid")
            if not filename.endswith('.mid'):
                filename += '.mid'
                
            filepath = f"handel_midi/{category}/{filename}"
            
            # Save the file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filepath}")
            return True
        else:
            print(f"Failed to download {url}: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

# Parse the Handel compositions page
def scrape_handel_page():
    base_url = "https://www.kunstderfuge.com/"
    handel_url = "https://www.kunstderfuge.com/handel.htm"
    
    try:
        # Get the main Handel page
        response = requests.get(handel_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links that might be MIDI files
        # This is a guess since we don't have the actual HTML structure
        links = soup.find_all('a')
        
        midi_count = 0
        max_daily_downloads = 5  # The site has a limit of 5 files per day
        
        downloaded_files = set()  # To avoid duplicates
        
        for link in links:
            if midi_count >= max_daily_downloads:
                print(f"Daily limit of {max_daily_downloads} MIDI files reached. Try again tomorrow or consider a subscription.")
                break
                
            href = link.get('href')
            if not href:
                continue
                
            # Look for links that might be MIDI files
            if href.endswith('.mid') or 'midi/' in href or 'midi.htm' in href:
                full_url = urljoin(base_url, href)
                
                # Skip if we've already processed this URL
                if full_url in downloaded_files:
                    continue
                
                # Get the title from link text and parent context
                title = link.text.strip()
                parent_text = link.parent.text.strip() if link.parent else ""
                
                # Extract description from surrounding text (this is a guess)
                description = ""
                if link.next_sibling and isinstance(link.next_sibling, str):
                    description = link.next_sibling.strip()
                elif link.parent and link.parent.next_sibling:
                    description = link.parent.next_sibling.strip()
                
                # If no good description, use surrounding text
                if not description and parent_text:
                    description = parent_text.replace(title, '').strip()
                
                # Default if nothing else works
                if not title:
                    title = f"handel_{midi_count+1}"
                if not description:
                    description = "unknown"
                
                # Download the file
                if download_midi(full_url, title, description):
                    midi_count += 1
                    downloaded_files.add(full_url)
                
                # Be nice to the server
                time.sleep(1)
        
        print(f"Downloaded {midi_count} MIDI files")
        
        # If we hit the limit, suggest what to do next
        if midi_count >= max_daily_downloads:
            print("\nTo download more files, you can:")
            print("1. Try again tomorrow (5 more files)")
            print("2. Subscribe to kunstderfuge.com for unlimited downloads")
            print("3. Modify the script to use a different IP or wait 24 hours")
    
    except Exception as e:
        print(f"Error scraping Handel page: {str(e)}")

if __name__ == "__main__":
    print("Starting Handel MIDI file scraper...")
    setup_directories()
    scrape_handel_page()
    print("Scraping complete.")