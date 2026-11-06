import zipfile, re, sys, csv, xml.etree.ElementTree as ET
ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
z = zipfile.ZipFile(sys.argv[1])
ss = [''.join(t.text or '' for t in si.iter('{%s}t' % ns['m'])) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
wb = ET.fromstring(z.read('xl/workbook.xml')); rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
rid = {r.get('Id'): r.get('Target') for r in rels}
sheet = [s for s in wb.find('m:sheets', ns) if s.get('name') == sys.argv[2]][0]
target = rid[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')].lstrip('/')
target = target if target.startswith('xl/') else 'xl/' + target
rows = []
def col(ref):
    c = re.match(r'[A-Z]+', ref).group(); n = 0
    for ch in c: n = n * 26 + ord(ch) - 64
    return n - 1
for r in ET.fromstring(z.read(target)).iter('{%s}row' % ns['m']):
    vals = {}
    for c in r.findall('m:c', ns):
        v = c.find('m:v', ns); t = c.get('t')
        if t == 'inlineStr':
            val = ''.join(x.text or '' for x in c.iter('{%s}t' % ns['m']))
        elif v is None: val = ''
        elif t == 's': val = ss[int(v.text)]
        else: val = v.text
        vals[col(c.get('r'))] = val
    rows.append([vals.get(i, '') for i in range(max(vals) + 1)] if vals else [])
w = max(len(r) for r in rows); csv.writer(open(sys.argv[3], 'w', newline='')).writerows([r + [''] * (w - len(r)) for r in rows])
