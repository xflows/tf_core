import os.path
import re
# from docx import opendocx, getdocumenttext # TODO update with python-docx
# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
# from io import StringIO
from subprocess import PIPE, Popen

# TODO Rewrite this part, pdfminer doesn't work under python 3
# def convert_pdf_to_txt(path):
#     rsrcmgr = PDFResourceManager()
#     retstr = StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = file(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ""
#     maxpages = 0
#     caching = True
#     pagenos = set()

#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
#         interpreter.process_page(page)
#     fp.close()
#     device.close()
#     str = retstr.getvalue()
#     retstr.close()
#     return str


def convert_pdf_to_txt(path):
    raise Exception("This part of the code is not yet ported to Python3")


def document_to_text(path):
    ext = os.path.splitext(path)[1]
    filename = os.path.basename(path)

    if ext == ".doc":
        cmd = os.path.dirname(os.path.realpath(
            __file__)) + os.sep + "antiword" + os.sep + "antiword.exe -m CP852 " + path
        pipe = Popen(cmd, stdout=PIPE, shell=True)
        text = pipe.communicate()[0]
        return filename, str(re.sub("\r|\n", "", text).strip())

    elif ext == ".docx":
        document = opendocx(path)
        paratextlist = getdocumenttext(document)
        newparatextlist = []
        for paratext in paratextlist:
            newparatextlist.append(paratext)
        return filename, "".join(newparatextlist)

    elif ext == ".pdf":
        text = convert_pdf_to_txt(path)
        return filename, str(re.sub("\r|\n", "", text).strip())

    else:
        text = open(path, "r").read().strip()
        return filename, text
