date=$(date '+%Y-%m-%d %H:%M:%S')
printf '\n'
#echo Starting $(date '+%Y-%m-%d %H:%M:%S') >>/usr/Newspaper-Scrapers/cron.txt
cd /usr/Newspaper-Scrapers/
/opt/anaconda3/bin/python webscrapper.py>>/usr/Newspaper-Scrapers/cron.txt
/opt/anaconda3/bin/python NewspaperScraper.py>>/usr/Newspaper-Scrapers/cron.txt
echo Completed $(date '+%Y-%m-%d %H:%M:%S') >>/usr/Newspaper-Scrapers/cron.txt
