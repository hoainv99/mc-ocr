from dateutil.parser import parse, ParserError
from calendar import IllegalMonthError
import re
def date_finder(string):
    regexr = re.search(r'\b((?:\d\d[-/\.:])+\d\d(\d\d)?)[\s-]?((?:\d\d[\.:])+\d\d)?\b', string)
#     print(regexr)
    if regexr is None: return None
#     try:
#         parse(regexr.group(1) or '' + ' ' + regexr.group(2) or '')
#     except:
#         return None
    return regexr.group(0), regexr.start(0), regexr.end(0)