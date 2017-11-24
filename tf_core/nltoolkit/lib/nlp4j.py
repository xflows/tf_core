import subprocess
import os

interpelations = {'NP':'NNP', 'NPS':'NNPS', 'PP':'PRP', 'PP$':'PRP$', 
                  'VD':'VB', 'VDD':'VBD', 'VDG':'VBG', 'VDN':'VBN', 'VDZ': 'VBZ', 'VDP':'VBP',
                  'VH':'VB', 'VHD':'VBD', 'VHG':'VBG', 'VHN':'VBN', 'VHZ': 'VBZ', 'VHP':'VBP',
                  'VV':'VB', 'VVD':'VBD', 'VVG':'VBG', 'VVN':'VBN', 'VVZ': 'VBZ', 'VVP':'VBP' }


class NLP4JTagger():

    def __init__(self, path_to_bin, config_file, input_path, train_path, pretrained=True):
        self.path_to_bin = path_to_bin
        self.config_file = config_file
        self.train_path = train_path
        self.input_path = input_path

    def train(self, path_to_bin, config_file, train_path, mode, output_model="", previous_model="", dev_path="", train_ext="*",
              dev_ext="*", cross_validation=0):
        args =[]
        args.append(os.path.join(path_to_bin, 'nlptrain.bat'))
        args.extend(["-c", config_file])
        args.extend(["-t", train_path])
        args.extend(["-mode", mode])
        args.extend(["-cv", str(cross_validation)])
        if output_model:
            args.extend(["-m", output_model])
        if previous_model:
            args.extend(["-p", previous_model])
        if dev_path:
            args.extend(["-d", dev_path])
        if train_ext:
            args.extend(["-te", train_ext])
        if dev_ext:
            args.extend(["-de", dev_ext])
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1)
        for line in iter(popen.stdout.readline, ""):
            print(line)
        popen.wait()


    def decode(self, sentences):
        input_ext="*" 
        utput_ext="nlp" 
        format="raw"
        threads=2
        
        if isinstance(sentences, list):
            new_sents = []
            for sentence in sentences:
                sentence = ' '.join(sentence)
                sentence = sentence.replace(" 's ", "'s ")
                sentence = sentence.replace(" 'm ", "'m ")
                sentence = sentence.replace(" 're ", "'re ")
                sentence = sentence.replace(" 'll ", "'ll ")
                sentence = sentence.replace(" 've ", "'ve ")
                sentence = sentence.replace(" 'd ", "'d ")
                #sentence = sentence.replace('\xe2\x84\x87"', "\xe2\x84\x87")
                #sentence = re.sub(r" ([^ ']+)' ", r" \1 ", sentence)
                #sentence = re.sub(r" '([^ ']+) ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ ']+)-- ", r" \1 ", sentence)
                #sentence = re.sub(r" ([^ .]+)\. ", r" \1 ", sentence)

                new_sents.append(sentence)

            sentences = new_sents
            sentences = ' ### '.join(sentences)
           
            f = open(self.input_path, 'w')
            f.write(sentences)
            f.close()
        args = []
        args.append(os.path.join(self.path_to_bin, 'nlpdecode.bat'))
        args.extend(["-c", self.config_file])
        args.extend(["-i", self.input_path])
        #args.extend(["-ie", input_ext])
        #args.extend(["-oe", output_ext])
        #args.extend(["-format", format])
        #args.extend(["-threads", str(threads)])
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, bufsize=1)
        popen.communicate()

        f = open(self.input_path + '.nlp', 'r')
        taggedSnt = f.read()
        f.close()
        
        os.remove(self.input_path)
        os.remove(self.input_path + '.nlp')

        sentences = []
        sentence = []
        splitted_tags = taggedSnt.split('\n')
        last = len(splitted_tags) - 1
        for i, line in enumerate(splitted_tags):
            line = line.strip()
            try:
                word = line.split('\t')[1].strip()
                tag = line.split('\t') [3].strip()
                print(word, tag)

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
        sentences.append(sentence)
        print(sentences)
