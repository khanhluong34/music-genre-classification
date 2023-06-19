import os

def get_genre_urls(genre_urls_path: str):
    genre_urls = {}
    genre_urls_list = os.listdir(genre_urls_path)

    for genre in genre_urls_list:
        urls = []
        path = os.path.join(genre_urls_path, genre)
        with open(path, 'r') as f:
            for line in f:
                urls.append(line.replace('\n', ''))
        genre_urls[genre.replace('.txt', "")] = urls
    return genre_urls


if __name__ == '__main__':
    genre_urls_path = './genre_urls'
    genre_urls = get_genre_urls(genre_urls_path)
    print(genre_urls.keys())