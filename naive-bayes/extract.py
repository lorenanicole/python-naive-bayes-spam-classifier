import email.parser
import lxml.html

def extract_body(filename):
    fp = open(filename)
    msg = email.message_from_file(fp)
    payload = msg.get_payload()
    if type(payload) == type(list()):
        payload = payload[0]
    plain_text_body_content = lxml.html.document_fromstring(str(payload)).text_content()
    return plain_text_body_content
