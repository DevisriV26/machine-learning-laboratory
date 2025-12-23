import csv, re
from collections import defaultdict
from datetime import datetime

VALID_ACTIVITIES = {"LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"}

class Student:
    def __init__(self, sid, name):
        self.sid, self.name, self.activities = sid, name, []
    def add_activity(self, act, date, time):
        self.activities.append((act, date, time))
    def summary(self):
        return sum(a=="LOGIN" for a,_,_ in self.activities), sum(a=="SUBMIT_ASSIGNMENT" for a,_,_ in self.activities)

def valid_id(sid): return bool(re.match(r"^S\d+$", sid))
def valid_act(act): return act in VALID_ACTIVITIES
def parse_dt(d, t): return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").date(), datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").time()

def read_log(file):
    with open(file) as f:
        for i, row in enumerate(csv.reader(f, delimiter="|"),1):
            try:
                if len(row)!=5: raise ValueError("Invalid row length")
                sid,name,act,d,t = [x.strip() for x in row]
                if not valid_id(sid) or not valid_act(act): raise ValueError("Invalid ID/activity")
                date,time = parse_dt(d,t)
                yield sid,name,act,date,time
            except Exception as e: print(f"Skipping line {i}: {e}")

def process_logs(infile, outfile):
    students,abnormal,daily = {},defaultdict(list),defaultdict(lambda: {"LOGIN":0,"LOGOUT":0,"SUBMIT_ASSIGNMENT":0})
    for sid,name,act,date,time in read_log(infile):
        if sid not in students: students[sid]=Student(sid,name)
        students[sid].add_activity(act,date,time)
        daily[date][act]+=1
        if act=="LOGIN":
            if abnormal[sid] and abnormal[sid][-1]=="LOGIN": print(f"Abnormal login: {sid} on {date} {time}")
            abnormal[sid].append("LOGIN")
        elif act=="LOGOUT": abnormal[sid].append("LOGOUT")

    with open(outfile,"w") as f:
        f.write("StudentID | Name | Total Logins | Total Submissions\n")
        print("StudentID | Name | Total Logins | Total Submissions")
        for s in students.values():
            l,sb = s.summary()
            line = f"{s.sid} | {s.name} | {l} | {sb}"
            print(line); f.write(line+"\n")

        f.write("\nDaily Activity Stats:\n")
        print("\nDaily Activity Stats:")
        for date, stats in daily.items():
            line = f"{date}: LOGIN={stats['LOGIN']}, LOGOUT={stats['LOGOUT']}, SUBMIT_ASSIGNMENT={stats['SUBMIT_ASSIGNMENT']}"
            print(line); f.write(line+"\n")

if __name__=="__main__":
    process_logs("student_logs.csv","student_report.txt")
