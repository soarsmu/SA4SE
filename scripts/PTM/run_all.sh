for i in 0 1 2 3
do
    python github.py -m $i
    python api.py -m $i
    python app.py -m $i
    python so.py -m $i
    python jira.py -m $i
    python cr.py -m $i
done