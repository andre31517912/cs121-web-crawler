import re
import urllib
from urllib.parse import urlparse
import spacy
from bs4 import BeautifulSoup
import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS

from collections import Counter


#a printer function that writes the tokenized words in a readable format
def printer(input_list):
    for token in input_list:
        print( f"{str(token.text_with_ws):22}" + f"{str(token.is_alpha):15}" + f"{str(token.is_punct):18}" + f"{str(token.is_stop)}")

#cleans the text and by substituting the special characters with ""
def text_cleaner(text):
    new_text = re.sub(r"[^\w\s]", "",text)
    new_text = new_text.lower()
    return text

#tokeniszer class --> uses the nltk library to produce tokens from the cleaned up text
def tokenizer(text):
    cur_tokens = nltk.word_tokenize(text)
    return cur_tokens

#this is the tokenizer we finalized on ------> 
#uses the spacy library, i preferred it to the nltk library because of 
def second_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")
    tokenized = nlp(text)
    return tokenized

#filters the urls by repairing urls that aren't in full path form
#it also makes sure that urls that don't have the http in front of it
#if its a broken url it discards it and returns the set of good urls
def filter_urls(base, urls):
    cur_filter = set()
    for z in urls:
        
        if z is None or z.strip() is None:
            continue
        if z.startswith("javascript"):
            continue
        if not z.startswith('http'):
            url = urljoin(base,z)

        
            cur_filter.add(url)
        else:
            cur_filter.add(z)
    return cur_filter


def download_url(urlpath):
    try:
        # open a connection to the server
        with urlopen(urlpath, timeout=3) as connection:
            # read the contents of the url as bytes and return it
            return connection.read()
    except:
        return None

#makes sure that the url actually has content and isnt an empty page
def filter1(urls):
    
    for i in urls:
        if download_url(i) is None:
            urls.remove(i)
    return urls


def scraper(url, resp, current_hashmap, currentwords):
    r = extract_next_links(url, resp,current_hashmap,currentwords)

    return r

def top_50_words(words_list):
    return {word:count for word, count in Counter(words_list).most_common(50)}

def extract_next_links(url, resp, current_hashmap, current_words):
    
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    # if url status is not 200 return empty list cuz the url shouldnt be used
    if resp.status != 200:
        return list()

    #this section of code finds out the links in the current url
    cur_soup =BeautifulSoup(resp.raw_response.content, 'html.parser')
    links = []
    for a in cur_soup.find_all('a', href=True):
        url = a['href']
        if(is_valid(a['href'])):
            
            links.append(a['href'])
    
    new_links = list(filter_urls(url,links))
    

    for z in new_links:
        if z in current_hashmap.keys():
            new_links.remove(z)
    #finds all the text and then makes sure to strip all excess spaces
    ps = [p.get_text().strip('\n') for p in cur_soup.find_all("p")]
    
    text = ""
    for i in ps:
        if i == " " or i == "":
            pass
        else:
            f = i.lstrip().rstrip().strip('\t')
            if f.endswith(".") or f.endswith(","):
                f+=" " 
            text += i

    
    #hashmap for the url
    f = tokenizer(text_cleaner(text)) # nltk tokenizer
    r = second_tokenizer(text) # spacy tokenizer
    #printer(r)

    
    #tokenized words is the list of tokens made by the second_tokenizer(test) call and is both not a stop word and is an alphanumeric word
    #stopwords is a list of words that contains all the stop words in the text
    #textual 
    tokenized_words = []
    stop_word = spacy.lang.en.stop_words.STOP_WORDS
    stopwords = []
    
    
    for i in r:

        if not i.is_stop and i.is_alpha:
            tokenized_words.append(i)
            current_words.append(i)
        elif i.is_stop:
            stopwords.append(i)

    #textual content is the ratio of tokenized words to the total number of words in the text
    if len(stopwords) + len(tokenized_words) == 0: return list()
    else: textual_content = len(tokenized_words)/(len(tokenized_words) + len(stopwords))
    #hashmap is key --> value 
    #key is the url
    #value is the ["raw text" --> all words in the url, links is a list of urls that are in the page, r is the nlp call on the tesxt, tokenized words, stop words, textual content percentage]
    
    #implement trap here --> before adding it into the hashmap

    if url not in current_hashmap.keys():
        current_hashmap[url] =[text,links,r,tokenized_words,stopwords, textual_content, [], textual_content,simhash(text)]
    
    print(ics_edu(list(current_hashmap.keys())))


    


    
    return [list(current_hashmap[url][1]), current_words, current_hashmap]


#finds the count of urls that have "ics.uci.edu" in

def ics_edu(current_urls):
    count =0
    l = []
    for z in current_urls:

        if "ics.uci.edu" in z:
            count+=1
            l.append(z)   
    return count


#implemented to find loops of urls

def find_traps(urls,all_urls):
    traps = []
    stack = []
    visited = set()
    for url in urls:
        if url not in visited:
            visited.add(url)
            stack.append(url)
            trap = [url]
            while stack:
                current = stack.pop()
                for neighbor in get_neighbors(current,all_urls):
                    if neighbor in visited:
                        trap.append(neighbor)
                    else:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(trap) > 1:
                traps.append(trap)
    return traps


def get_neighbors(url, all_urls):
    # Example implementation: return all URLs that have the same domain as the input URL
    return [u for u in all_urls if u.split(".")[-2:] == url.split(".")[-2:]]

def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        parsed = urlparse(url)
        if parsed.scheme not in set(["http", "https"]):
            return False
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower())

        if not ( "https://www.ics.uci.edu" in url or "https://www.cs.uci.edu" in url  or "https://www.informatics.uci.edu" in url or "https://www.stat.uci.edu" in url):
            return False
    except TypeError:
        print ("TypeError for ", parsed)
        raise

#takes in input of the text and the tokenized words and performs a simhash on the number

def simhash(text):
    words = text.split()
    hash_list = [sum([ord(c) for c in word]) for word in words]
    simhash = [0] * 32
    for h in hash_list:
        for i in range(32):
            simhash[i] += (h & 1) * 2 - 1
            h >>= 1
    for i in range(32):
        if simhash[i] > 0:
            simhash[i] = 1
        else:
            simhash[i] = 0
    return int("".join(str(x) for x in simhash), 2)

#calculates the hamming distance between the two hashes
def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count("1")

#finds the similarity between the two texts by running the hamming distance on the two simhashes
def similarity(text1, text2):
    hash1 = simhash(text1)
    hash2 = simhash(text2)
    return hamming_distance(hash1, hash2)

def final_processing(current_hashmap):
    return max(current_hashmap.keys(), key=lambda x: len(current_hashmap[x][0]))
    
