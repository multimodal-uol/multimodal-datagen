from newspaper.newspaper import build
cnn_paper = build('http://cnn.com')
for article in cnn_paper.articles:
	print(article.url)

