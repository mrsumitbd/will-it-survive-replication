"""xlsx_read.py: read one sheet of an .xlsx file into a pandas DataFrame with the standard library only
(no openpyxl needed). First row = header."""
import re, zipfile, xml.etree.ElementTree as ET
import pandas as pd

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'


def _col(ref):
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path, sheet=None):
    z = zipfile.ZipFile(path)
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        ss = [''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    sheets = list(wb.find('m:sheets', NS))
    sh = sheets[0] if sheet is None else next(s for s in sheets if s.get('name') == sheet)
    target = rels[sh.get(RID)].lstrip('/')
    target = target if target.startswith('xl/') else 'xl/' + target
    rows = []
    for r in ET.fromstring(z.read(target)).iter('{%s}row' % NS['m']):
        vals = {}
        for c in r.findall('m:c', NS):
            v, t = c.find('m:v', NS), c.get('t')
            if t == 'inlineStr':
                val = ''.join(x.text or '' for x in c.iter('{%s}t' % NS['m']))
            elif v is None:
                val = ''
            elif t == 's':
                val = ss[int(v.text)]
            else:
                val = v.text
            vals[_col(c.get('r'))] = val
        rows.append([vals.get(i, '') for i in range(max(vals) + 1)] if vals else [])
    w = max(len(r) for r in rows)
    rows = [r + [''] * (w - len(r)) for r in rows]
    return pd.DataFrame(rows[1:], columns=rows[0])
