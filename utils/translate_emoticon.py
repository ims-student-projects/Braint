
"""
Emoticons to label mapper. List of emoticons is based on a list by EmojiOne
Labs (https://demos.emojione.com/ascii-smileys.html). The emoticons are in
regex representation, whereby potentially repeatable symbols are taken into
account, i.e. ":)" and ":))))))" both map to the same label "joy".

"""

emoticon_to_label = {
    r"<3+" : ":heart:",    # <3
    r"</3+" : ":broken_heart:",    # </3
    r":'\)+" : ":joy:",    # :')
    r":'-\)+" : ":joy:",    # :'-)
    r"\^\^" : ":joy:",    # ^^
    r"\^_\^" : ":joy:",    # ^_^
    r":D" : ":smile:",    # :D
    r":-D" : ":smile:",    # :-D
    r"=d" : ":smile:",    # =d
    r":\)+" : ":slight_smile:",    # :)
    r":-\)+" : ":slight_smile:",    # :-)
    r"=]" : ":slight_smile:",    # =]
    r"=\)+" : ":slight_smile:",    # =)
    r":]" : ":slight_smile:",    # :]
    r"':\)+" : ":sweat_smile:",    # ':)
    r"':-\)+" : ":sweat_smile:",    # ':-)
    r"'=\)+" : ":sweat_smile:",    # '=)
    r"':D" : ":sweat_smile:",    # ':D
    r"':-D" : ":sweat_smile:",    # ':-D
    r"'=D" : ":sweat_smile:",    # '=D
    r">:\)+" : ":laughing:",    # >:)
    r">;\)+" : ":laughing:",    # >;)
    r">:-\)+" : ":laughing:",    # >:-)
    r">=\)+" : ":laughing:",    # >=)
    r";\)+" : ":wink:",    # ;)
    r";-\)+" : ":wink:",    # ;-)
    r"\*-\)+" : ":wink:",    # *-)
    r"\*\)+" : ":wink:",    # *)
    r";-]" : ":wink:",    # ;-]
    r";]" : ":wink:",    # ;]
    r";D" : ":wink:",    # ;D
    r";\^\)+" : ":wink:",    # ;^)
    r"':\(" : ":sweat:",    # ':(
    r"':-\(" : ":sweat:",    # ':-(
    r"'=\(" : ":sweat:",    # '=(
    r":\*" : ":kissing_heart:",    # :*
    r":-\*" : ":kissing_heart:",    # :-*
    r"=\*" : ":kissing_heart:",    # =*
    r":\^\*" : ":kissing_heart:",    # :^*
    r">:P" : ":stuck_out_tongue_winking_eye:",    # >:P
    r"X-P" : ":stuck_out_tongue_winking_eye:",    # X-P
    r"x-p" : ":stuck_out_tongue_winking_eye:",    # x-p
    r">:\[" : ":disappointed:",    # >:[
    r":-\(" : ":disappointed:",    # :-(
    r":\(" : ":disappointed:",    # :(
    r":-\[" : ":disappointed:",    # :-[
    r":\[" : ":disappointed:",    # :[
    r"=\(\)+" : ":disappointed:",    # =()
    r">:\(" : ":angry:",    # >:(
    r">:-\(" : ":angry:",    # >:-(
    r":@" : ":angry:",    # :@
    r":'\(" : ":cry:",    # :'(
    r":'-\(" : ":cry:",    # :'-(
    r";\(" : ":cry:",    # ;(
    r";-\(" : ":cry:",    # ;-(
    r">.<" : ":persevere:",    # >.<
    r"D:" : ":fearful:",    # D:
    r":\$" : ":flushed:",    # :$
    r"=\$" : ":flushed:",    # =$
    r"#-\)+" : ":dizzy_face:",    # #-)
    r"#\)+" : ":dizzy_face:",    # #)
    r"%-\)+" : ":dizzy_face:",    # %-)
    r"%\)+" : ":dizzy_face:",    # %)
    r"X\)+" : ":dizzy_face:",    # X)
    r"X-\)+" : ":dizzy_face:",    # X-)
    r"\*\0/*" : ":ok_woman:",    # *\0/*
    r"\0/" : ":ok_woman:",    # \0/
    r"\*\\O/\*" : ":ok_woman:",    # *\O/*
    r"\\O/" : ":ok_woman:",    # \O/
    r"O:-\)+" : ":innocent:",    # O:-)
    r"0:-3" : ":innocent:",    # 0:-3
    r"0:3" : ":innocent:",    # 0:3
    r"0:-\)+" : ":innocent:",    # 0:-)
    r"0:\)+" : ":innocent:",    # 0:)
    r"0;\^\)+" : ":innocent:",    # 0;^)
    r"O:-\)+" : ":innocent:",    # O:-)
    r"O:\)+" : ":innocent:",    # O:)
    r"O;-\)+" : ":innocent:",    # O;-)
    r"O=\)+" : ":innocent:",    # O=)
    r"0;-\)+" : ":innocent:",    # 0;-)
    r"O:-3" : ":innocent:",    # O:-3
    r"O:3" : ":innocent:",    # O:3
    r"B-\)+" : ":sunglasses:",    # B-)
    r"B\)+" : ":sunglasses:",    # B)
    r"8\)+" : ":sunglasses:",    # 8)
    r"8-\)+" : ":sunglasses:",    # 8-)
    r"B-D" : ":sunglasses:",    # B-D
    r"8-D" : ":sunglasses:",    # 8-D
    r"-_+-" : ":expressionless:",    # -_-
    r">:\\" : ":confused:",    # >:\
    r":P" : ":stuck_out_tongue:",    # >:/
    r":-P" : ":stuck_out_tongue:",    # :-/
    r"=p" : ":stuck_out_tongue:",    # :-.
    r":-p" : ":stuck_out_tongue:",    # :/
    r":p" : ":stuck_out_tongue:",    # :\
    r"=p" : ":stuck_out_tongue:",    # =/
    r":-Þ" : ":stuck_out_tongue:",    # =\
    r":Þ" : ":stuck_out_tongue:",    # :L
    r":þ" : ":stuck_out_tongue:",    # =l
    r":-þ" : ":stuck_out_tongue:",    # :P
    r":-b" : ":stuck_out_tongue:",    # :-P
    r":b" : ":stuck_out_tongue:",    # =p
    r"d:" : ":stuck_out_tongue:",    # :-p
    r":-O" : ":open_mouth:",    # :p
    r":O" : ":open_mouth:",    # =p
    r":-o" : ":open_mouth:",    # :-Þ
    r":o" : ":open_mouth:",    # :Þ
    r"O_O" : ":open_mouth:",    # :þ
    r">:O" : ":open_mouth:",    # :-þ
    r":-X" : ":no_mouth:",    # :-b
    r":X" : ":no_mouth:",    # :b
    r":-#" : ":no_mouth:",    # d:
    r":#" : ":no_mouth:",    # :-O
    r"=x" : ":no_mouth:",    # :O
    r"=x" : ":no_mouth:",    # :-o
    r":x" : ":no_mouth:",    # :o
    r":-x" : ":no_mouth:",    # O_O
    r"=#" : ":no_mouth:",    # >:O
    ":-X" : ":no_mouth:",    # :-X
    ":X" : ":no_mouth:",    # :X
    ":-#" : ":no_mouth:",    # :-#
    ":#" : ":no_mouth:",    # :#
    "=X" : ":no_mouth:",    # =X
    "=x" : ":no_mouth:",    # =x
    ":x" : ":no_mouth:",    # :x
    ":-x" : ":no_mouth:",    # :-x
    "=#" : ":no_mouth:",    # =#
        }


def translate_emoticon(emoticon):
    return emoticon_to_label[emoticon]
