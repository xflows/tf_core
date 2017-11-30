import subprocess
import os
import xml.etree.cElementTree as ET
import csv

interpelations = {'NP':'NNP', 'NPS':'NNPS', 'PP':'PRP', 'PP$':'PRP$', 
                  'VD':'VB', 'VDD':'VBD', 'VDG':'VBG', 'VDN':'VBN', 'VDZ': 'VBZ', 'VDP':'VBP',
                  'VH':'VB', 'VHD':'VBD', 'VHG':'VBG', 'VHN':'VBN', 'VHZ': 'VBZ', 'VHP':'VBP',
                  'VV':'VB', 'VVD':'VBD', 'VVG':'VBG', 'VVN':'VBN', 'VVZ': 'VBZ', 'VVP':'VBP' }


class NLP4JTagger():

    def __init__(self, path_to_bin, config_file, input_path, train_path, dev_path, model_path, train_config_file, pretrained=True):
        self.path_to_bin = path_to_bin
        self.config_file = config_file
        self.train_path = train_path
        self.dev_path = dev_path
        self.input_path = input_path
        self.model_path = model_path
        self.train_config_file = train_config_file

    def train(self, sentences, mode="pos"):

        if os.path.exists(self.train_path):
            os.remove(self.train_path)
      
        with open(self.train_path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            for sentence in sentences:
                for word, tag in sentence:
                     writer.writerow((word, tag))

        if os.path.exists(self.dev_path):
            os.remove(self.dev_path)

        with open(self.dev_path, "wb") as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            for sentence in sentences:
                for word, tag in sentence:
                     writer.writerow((word, tag))

        self.generate_xml()

        args =[]
        args.append(os.path.join(self.path_to_bin, 'nlptrain'))
        args.extend(["-c", self.train_config_file])
        args.extend(["-t", self.train_path])
        args.extend(["-mode", mode])
        args.extend(["-m", self.model_path])
        args.extend(["-d", self.dev_path])
       
        print(args)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1)
        for line in iter(popen.stdout.readline, ""):
            print(line)
        popen.communicate()


    def decode(self, sentences):
        input_ext="*" 
        utput_ext="nlp" 
        format="raw"
        threads=2

        old = sentences

        if isinstance(sentences, list):
            if os.path.exists(self.input_path):
                os.remove(self.input_path)
            f = open(self.input_path, 'a')
            for sentence in sentences:
                #sentence = ' '.join(sentence)
                #sentence = sentence.replace(" 's ", "'s ")
                #sentence = sentence.replace(" 'm ", "'m ")
                #sentence = sentence.replace(" 're ", "'re ")
                #sentence = sentence.replace(" 'll ", "'ll ")
                #sentence = sentence.replace(" 've ", "'ve ")
                #sentence = sentence.replace(" 'd ", "'d ")
                #sentence = sentence.replace('\xe2\x84\x87"', "\xe2\x84\x87")
                #sentence = re.sub(r" ([^ ']+)' ", r" \1 ", sentence)
                #sentence = re.sub(r" '([^ ']+) ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ ']+)-- ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ .]+)\. ", r" \1 ", sentence)
                for token in sentence:
                    if len(token) == 0:
                        f.write(' -NONE- ')
                    elif token == '###':
                        f.write(' # ')
                    else:
                        f.write(" " + token + " ")
                f.write(' ### ') 
            f.close()
           
        args = []
        args.append(os.path.join(self.path_to_bin, 'nlpdecode'))
        args.extend(["-c", self.config_file])
        args.extend(["-i", self.input_path])
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1)
        popen.communicate()

        f = open(self.input_path + '.nlp', 'r')
        taggedSnt = f.read()
        f.close()

        
        os.remove(self.input_path)
        os.remove(self.input_path + '.nlp')

        sentences = []
        sentence = []
        splitted_tags = taggedSnt.split('\n')[:-1]
        last = len(splitted_tags) - 1
        for i, line in enumerate(splitted_tags):
            line = line.strip()
            try:
                word = line.split('\t')[1].strip()
                tag = line.split('\t') [3].strip()

                if word == '###':
                    sentences.append(sentence)
                    sentence = []
                else:
                    if tag == 'SENT':
                        sentence.append((word, '.'))
                    else:
                        if tag in interpelations:
                            sentence.append((word, interpelations[tag]))
                        else:
                            sentence.append((word, tag))
            except:
                pass
        
        

        checked_sentences = []
        for i, sentence in enumerate(sentences):
            if len(old[i]) != len(sentence):
                sentence = self.fixTokens(old[i], sentence)
            checked_sentences.append(sentence)
        return checked_sentences


    def fixTokens(self, standard, new):
        standard = [u"#" if t==u"###" else t for t in standard]
        fixed = []
        fix = 0
        new_tag = ""
        messedup_sent = False
        if len(standard) < len(new) or messedup_sent:
            new_token=""
            for i, token in enumerate(new):
                word = token[0]
                tag = token[1]
                if word == standard[fix] and new_token=="":
                    fix += 1
                    fixed.append(token)
                else:
                    revert_token = new_token
                    old_token = new_token
                    new_token += word
                    if not standard[fix].startswith(new_token):
                        old_token += " " + word
                        new_token = old_token
                    if not standard[fix].startswith(new_token):
                        #the most complicated scenario with completely wrong tokens
                        messedup_sent = True
                        for j in range(i, len(new)):
                            fixed.append(new[j])
                        break
                    else:
                        if len(new_tag) == 0 or len(word) > 1:
                            new_tag = tag
                        if new_token == standard[fix]:
                            fixed.append((new_token, tag))
                            fix += 1
                            new_token = ""
                            new_tag = ""
            #if not len(fixed) == len(standard):
            #    print('ni popravu: ', standard, new, fixed)

            new = fixed
        fixed = []
        fix = 0
        new_tag = ""
        if len(standard) > len(new) or messedup_sent:
            sliced = ""
            for i, token in enumerate(standard):
                word = token
                predicted_token = new[fix][0]
                if word == predicted_token:
                    fixed.append((token, new[fix][1]))
                    fix += 1
                else:
                    if sliced == "":
                        sliced = predicted_token
                    fixed.append((word, new[fix][1]))
                    sliced = sliced[len(word):]
                    if sliced == "":
                        fix += 1
                    
            #if not len(fixed) == len(standard):
            #    print('ni popravu dolzga: ', standard, fixed)
            #else:
            #    print('popravu', standard, fixed)
            new = fixed

        if not len(new) == len(standard):
            print('ni popravu: ', standard, new)

        return new

    def generate_xml(self):
        root = ET.Element("configuration")
        tsv = ET.SubElement(root, "tsv")
        lexica =  ET.SubElement(root, "lexica")
        models = ET.SubElement(root, "models")

        ET.SubElement(tsv, "column", index="1", field="form")
        ET.SubElement(lexica, "ambiguity_classes", field="word_form_simplified_lowercase").text = "edu/emory/mathcs/nlp/lexica/en-ambiguity-classes-simplified-lowercase.xz"
        ET.SubElement(lexica, "word_clusters", field="word_form_simplified_lowercase").text = "edu/emory/mathcs/nlp/lexica/en-brown-clusters-simplified-lowercase.xz"
        ET.SubElement(models, "pos").text = self.model_path

        
        tree = ET.ElementTree(root)
        tree.write(self.config_file)