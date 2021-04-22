import twint
import sys

def keywordsearch(since, until, output, term, near,limit):
    c = twint.Config()
    c.Since = since
    c.Until = until
    c.Store_json = True
    c.Output = output
    c.Search = term
    if near != 'all':
        c.Near = near
    if limit != 'all':
        c.Limit = limit
    tweets = twint.run.Search(c)
    return tweets

def usersearch(since, until, output, user, near,limit):
    c = twint.Config()
    c.Since = since
    c.Until = until
    c.Store_json = True
    c.Output = output
    c.Username = user
    c.Lang='en'
    if near != 'all':
        c.Near = near
    if limit != 'all':
        c.Limit = limit
    tweets = twint.run.Search(c)
    return tweets
    
since = sys.argv[1]
until = sys.argv[2]
output = sys.argv[3]
term = sys.argv[4]
near = sys.argv[5]
user = sys.argv[6]
limit = sys.argv[7]
if user == 'null':
    keywordsearch (since, until, output, term,near,limit)
else:
    usersearch (since, until, output, user,near,limit)

