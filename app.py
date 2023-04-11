from flask import Flask,render_template,request
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sum',methods=['POST','GET'],)
def sum():
    body=request.form['data']
    sm=request.form['sumerizer']


    if sm == "POS":

 #preprocessing       
        import nltk
        import torch
        import string
        nltk.download('punkt')
        #nltk.download('all')
        nltk.download('averaged_perceptron_tagger')
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        from nltk.tokenize import word_tokenize,sent_tokenize
        from summarizer import Summarizer



        #nltk.download()

        text = body

        model = Summarizer()
        result = model(text, min_length=60)
        full = ''.join(result)



        summary = result

        text_tokens = word_tokenize(summary)

        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

        filtered_sentence = (" ").join(tokens_without_sw)


        all_stopwords = stopwords.words('english')
        all_stopwords.append('play')

        text_tokens = word_tokenize(summary)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]

        sw_list = ['»','x0eA']
        all_stopwords.extend(sw_list)

        text_tokens = word_tokenize(summary)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]

        filtered_sentence = (" ").join(tokens_without_sw)

        #import string

        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(filtered_sentence)

        tokens = list(filter(lambda token: token not in string.punctuation, tokens))

        string = " ".join( tokens ) 

#pos
        text = string
        stop_words = set(stopwords.words('english'))

        # Tokenize the text
        tokens = sent_tokenize(text)
        #print(tokens)

        for i in tokens:
            #print(i)
            words = word_tokenize(i)
            words = [w for w in words if not w in stop_words]
            #print(words)
            # pos tagger
            tags = nltk.pos_tag(words)

        def tuple_to_list(t):
            new_list = []
            for element in t:
                new_list.append(element)
            return new_list

        tuple_1 = tags
        pos_list=tuple_to_list(tuple_1)

        global s1
        s1 =[]
        for i in range(len(pos_list)):
            if(pos_list[i][1]=='NNP'):
                s1.append(pos_list[i][0])
#print(s1)
        global s2
        s2=[]
        for i in range(len(pos_list)):
            if(pos_list[i][1]=='VB' or pos_list[i][1]=='VBD' or pos_list[i][1]=='VBG' or pos_list[i][1]=='VBN'):
                s2.append(pos_list[i][0])   
        print(s2)

        #b = body.rsplit(".")
        #b1= body.rsplit(".")

        new = text

        #for i in text:
            #new = new + i
        
        fill = ""

        for i in s1:
            fill=new.replace(i,"_____________")
            new=fill

        


        return render_template('sum.html',body = fill,sm = sm,answer= s1)

#NER

    elif sm =="NER":
        from nltk.tag.stanford import StanfordNERTagger
        jar = "stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"
        model = "stanford-ner-2015-04-20/classifiers/"
        st_4class = StanfordNERTagger(model + "english.conll.4class.distsim.crf.ser.gz", jar, encoding='utf8')

        example_document = body

        st_4class.tag(example_document.split())

        import spacy
        import spacy.cli 
        spacy.cli.download("en_core_web_sm")

        sp_sm = spacy.load('en_core_web_sm')

        def spacy_large_ner(document):
            return {(ent.text.strip(), ent.label_) for ent in sp_sm(document).ents}
        
        spacy_large_ner(example_document)

        def tuple_to_list(t):
            new_list = []
            for element in t:
                new_list.append(list(element))
            return new_list

        tuple_1 = spacy_large_ner(example_document)
        ner_list=tuple_to_list(tuple_1)
        #print(ner_list)

        #from __future__import print_function

        global nw1
        nw1= []
        for i in range(len(ner_list)):
            if(ner_list[i][1]=='MONEY' or ner_list[i][1]=='ORG' or ner_list[i][1]=='GPE' or ner_list[i][1]=='ORDINAL' or ner_list[i][1]=='DATE' or ner_list[i][1]=='TIME'):
                nw1.append(ner_list[i][0])
        #print(nw1)

        f = example_document.rsplit(".")
        f1 = example_document.rsplit(".")

        for i in nw1:
            y=example_document.replace(i,"_____________")
            example_document=y

        #print(y)

        b = y
        return render_template('sum.html',body = b,sm = sm,answer = nw1)
    
@app.route('/evaluate',methods=['POST','GET'])
def evaluate():
    ans=request.form['ansr']
    summe = request.form['summe']

    import spacy
    nlp = spacy.load('en_core_web_md')
    
    ans = ans.split(",")
    j=0
    v=[]
    for i in s2:
        words=""
        words= ans[j].lower()+" "
        words= words + i.lower()
        tokens = nlp(words)
  
        token1, token2 = tokens[0], tokens[1]
        similar = token1.similarity(token2)
        v.append(token1.similarity(token2))
        j=j+1

    score=0
    for i in v:
        score=score+i


    return render_template('evaluate.html',a = s2,v=ans, t = v) 
                





if __name__ == '__main__':
    app.run(debug = True,port =8000 )