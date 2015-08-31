import requests
from bs4 import BeautifulSoup
import string
import re




def scrape(text_file, site_url):

    """ Scrapes sonnet content from http://www.shakespeares-sonnets.com
        and outputs to sonnets.txt file.

    Args:
        text_file: output text file for scraped sonnets
        site_url: url of site to be scraped

    Returns:
        A list of lists of sonnet sentences with each index corresponding
        to a sonnet (Used for word tokenization later)
    """

    all_sonnets = []

    for x in range(1, 154):
        sonnet_url = site_url + str(x)
        soup = BeautifulSoup(requests.get(sonnet_url).text)
        sonnet = []
        text_file.write("SONNET " + str(x) + "\n\n")
        for i in soup.find_all('p')[0].children:
            if i.find("em") == None:
                if i.get_text() != u'':
                    line = re.sub(r'[\r\n]', '', i.get_text())
                    sonnet.append(line)

        # Formatting for sonnet outputs in sonnet.txt
        for line in sonnet:
            text_file.write(line.encode('utf-8'))
            text_file.write("\n")

        text_file.write("\n\n")

        all_sonnets.append(sonnet)


    return all_sonnets




if __name__ == "__main__":


    text_file = open("sonnets.txt", "w")
    site_url = "http://www.shakespeares-sonnets.com/sonnet/"
    scrape(text_file, site_url)

