import newspaper
cnn_paper = newspaper.build('https://www.theguardian.com/uk')
for article in cnn_paper.articles:
    print(article.url)
