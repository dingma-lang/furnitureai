"""Bounded standard-library iterator for top-level COCO JSON arrays."""
import json

def reject_constant(value):raise ValueError('Nonstandard JSON constant '+value)

def entries(path,chunk=65536,max_value=8388608):
 with open(path,encoding='utf-8') as f:
  buffer='';position=0;eof=False;decoder=json.JSONDecoder(parse_constant=reject_constant);keys=set()
  def fill():
   nonlocal buffer,position,eof
   buffer=buffer[position:];position=0
   data=f.read(chunk);buffer+=data
   if not data:eof=True
  def peek():
   nonlocal position
   while True:
    while position<len(buffer) and buffer[position].isspace():position+=1
    if position<len(buffer):return buffer[position]
    if eof:return ''
    fill()
  def consume(char):
   nonlocal position
   if peek()!=char:raise ValueError('Unexpected JSON delimiter: expected '+char)
   position+=1
  def value():
   nonlocal position
   peek()
   while True:
    try:v,end=decoder.raw_decode(buffer,position)
    except json.JSONDecodeError:
     if eof or len(buffer)-position>max_value:raise ValueError('Invalid/oversize metadata JSON value')
     fill();continue
    # A number can end at a chunk boundary before further digits arrive.
    numeric_continuation=isinstance(v,(int,float)) and not isinstance(v,bool) and buffer[end:end+1] in ('.','e','E','+','-','0','1','2','3','4','5','6','7','8','9')
    if not eof and (end==len(buffer) or numeric_continuation):fill();continue
    if end-position>max_value:raise ValueError('Metadata entry exceeds bound')
    position=end;return v
  consume('{')
  while peek()!='}':
   key=value()
   if not isinstance(key,str) or key in keys:raise ValueError('Duplicate/invalid top-level key')
   keys.add(key);consume(':')
   if key in ('images','annotations','categories','licenses'):
    consume('[')
    while peek()!=']':
     yield key,value()
     if peek()==']':break
     consume(',')
     if peek()==']':raise ValueError('Trailing array comma')
    consume(']')
   else:yield key,value()
   if peek()=='}':break
   consume(',')
   if peek()=='}':raise ValueError('Trailing object comma')
  consume('}')
  if peek():raise ValueError('Trailing JSON data')
