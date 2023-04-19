from flask import Flask,render_template,request
from fileinput import filename
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/sum',methods=['POST','GET'])
def sum():


    body = request.form['data']

    global sm
    sm=request.form['sumerizer']


    from summarizer import Summarizer



        #nltk.download()

    text = body

    global result

    model = Summarizer()
    result = model(text, min_length=60)
    full = ''.join(result)



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

        #global s1
        #s1 =[]
        #for i in range(len(pos_list)):
            #if(pos_list[i][1]=='NNP'):
               # s1.append(pos_list[i][0])
#print(s1)
        global s2
        s2=[]
        for i in range(len(pos_list)):
            if(pos_list[i][1]=='VB' or pos_list[i][1]=='VBD' or pos_list[i][1]=='VBG' or pos_list[i][1]=='VBN'):
                s2.append(pos_list[i][0])   
        print(s2)

        b = summary.rsplit(".")
        #b1= body.rsplit(".")

        #new = text

        #for i in text:
            #new = new + i
        
        #fill = ""

        #for i in s2:
            #fill=b.replace(i,"_____________")
            #new=fill

        X=[]
        for i in s2:
            X.append(i)

        z=0
        c1=1
        check_list2=[]
        for i in b:
            cn=0
            k=[]
            k=i.split()
            ind=0
            for j in X:
                if(k.count(j)>0 and k.index(j)>ind):
                    ind=k.index(j)
                    check_list2.append(j)
                    k[ind]="______"
                    cn=cn+1
                elif(k.count(j+",")>0):
                    ind=k.index(j+",")
                    check_list2.append(j)
                    k[ind]="______,"
                    cn=cn+1
                else:
                    break
            for cu in range(cn):
                X.pop(0)
            a=""
            for j in k:
                if(j=="______"):
                    a=a+"("+str(c1)+")"
                    c1=c1+1
                elif(j=="______,"):
                    a=a+"("+str(c1)+")"
                    c1=c1+1
                a=a+" "+j
            a=a+"."  
            b[z]=a
            z=z+1
        
        fillin=""
        for i in  range(len(b)-1):
            fillin= fillin+b[i]


        return render_template('sum.html',body = fillin,sm = sm)
    
    if sm == "POS_N":

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
        
        print(s1)


        b = summary.rsplit(".")


        Y = s1
        z=0
        c=1
        check_list1=[]
        for i in b:
            k=[]
            k=i.split()
            for j in Y:
                if(k.count(j)>0):
                    #print(j)
                    ind=k.index(j)
                    check_list1.append(j)
                    k[ind]="______"
                elif(k.count(j+",")>0):
                    #print(j)
                    ind=k.index(j+",")
                    check_list1.append(j)
                    k[ind]="______,"
            a=""
            for j in k:
                if(j=="______"):
                    a=a+"("+str(c)+")"
                    c=c+1
                elif(j=="______,"):
                    a=a+"("+str(c)+")"
                    c=c+1
                a=a+" "+j
            a=a+"."  
            b[z]=a
            z=z+1

        
        fillin=""
        for i in  range(len(b)-1):
            fillin= fillin+b[i]


        return render_template('sum.html',body = fillin,sm = sm)

#NER

    elif sm =="NER":
        from nltk.tag.stanford import StanfordNERTagger
        jar = "stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"
        model = "stanford-ner-2015-04-20/classifiers/"
        st_4class = StanfordNERTagger(model + "english.conll.4class.distsim.crf.ser.gz", jar, encoding='utf8')

        

        example_document = full

        st_4class.tag(example_document.split())

        import spacy
        import spacy.cli 
        #spacy.cli.download("en_core_web_trf")

        sp_sm = spacy.load('en_core_web_trf')

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
        print(nw1)

        b= example_document.rsplit(".")
        f1 = example_document.rsplit(".")

        global y
        co=1
        for i in nw1:
            y=example_document.replace(i,"_____________"+"("+str(co)+")",1)
            co=co+1
        #y=re.sub(ri,"_________",example_document,1)
            example_document=y

        print(y)

        #print(full)

            
        return render_template('sum.html',body = y,sm = sm)
    
@app.route('/evaluate',methods=['POST','GET'])
def evaluate():
    ans=request.form['ansr']
    #summe = request.form['summe']


    if sm == "POS":
        import spacy
        nlp = spacy.load('en_core_web_md')
        
        ans = ans.split(",")
        print(ans)
        print(s2)
        j=0
        v=[]
        for i in s2:
            words=""
            words= ans[j].lower()+" "
            words= words + i.lower()
            tokens = nlp(words)
            token1, token2 = tokens[0], tokens[1]
            similar = token1.similarity(token2)


            print("Similarity:", token1.similarity(token2))
            v.append(similar)
            print(similar)
            j=j+1


        print(v)
        score=0
        for i in v:
            score=score+i
        score = int(score*10)
        print(score)
        print(s2)

        return render_template('evaluate.html',a = s2,v=ans,t = score,g = len(v)) 
    
    elif sm == "POS_N":
        import spacy
        nlp = spacy.load('en_core_web_md')
        
        ans = ans.split(",")
        j=0
        v=[]
        for i in s1:
            words=""
            words= ans[j].lower()+" "
            words= words + i.lower()
            tokens = nlp(words)
            token1, token2 = tokens[0], tokens[1]
            similar = token1.similarity(token2)


            print("Similarity:", token1.similarity(token2))
            v.append(similar)
            j=j+1

        score=0
        for i in v:
            score=score+i
        score = int(score*10)
        print(score)
        print(s1)

        return render_template('evaluate.html',a = s1,v=ans,t = score,g = len(v)) 
    
    
    elif sm == "NER":

        import spacy
        nlp = spacy.load('en_core_web_md')
        
        ans = ans.split(",")
        az=y.split("_____________")
        bz=[]
        az.pop(0)
        for i in az:
            
            bz.append(int(i[1]))

        print(bz)


        j=0
        v=[]
        for i in ans:
            w=""
            w1=""
            w= nlp(nw1[j].lower())
            w1= nlp(i.lower())
  
            print("Similarity:", w1.similarity(w))
            print(w,w1)
            v.append(w1.similarity(w))
            j=j+1

        score = 0
        for i in v:
            score = score +i
        score = int(score*10)
        print(score)
        
        return render_template('evaluate.html',a = nw1,v=ans, t = score,g = len(v)) 
    
@app.route('/qsgenerate')
def qsgenerate():
    return render_template('qsgenerate.html')


@app.route('/qsandansr',methods=['POST','GET'])
def qsandansr():    


    bdy=request.form['dataa']
    qs=request.form['QS']

    q = int(qs)

    from summarizer import Summarizer

    text = bdy
    model = Summarizer()
    result = model(text, min_length=60)
    full = ''.join(result)
    
    import nltk
    import numpy as np 
    import nltk as nlp
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    data = full
    noOfQues=qs

    class SubjectiveTest:

        def __init__(self, data, noOfQues):

            self.question_pattern = [
                "Explain in detail about ",
                "Define about ",
                "Write a short note on ",
                "What is known as ",
                "What do you mean by ",
                "Describe about "
            ]

            self.grammar = r"""
                CHUNK: {<NN>+<IN|DT>*<NN>+}
                {<NN>+<IN|DT>*<NNP>+}
                {<NNP>+<NNS>*}
            """
            
            self.summary = data
            self.noOfQues = noOfQues
        
        @staticmethod
        def word_tokenizer(sequence):
            word_tokens = list()
            for sent in nlp.sent_tokenize(sequence):
                for w in nlp.word_tokenize(sent):
                    word_tokens.append(w)
            return word_tokens
        
        def create_vector(answer_tokens, tokens):
            return np.array([1 if tok in answer_tokens else 0 for tok in tokens])
        
        def cosine_similarity_score(vector1, vector2):
            def vector_value(vector):
                return np.sqrt(np.sum(np.square(vector)))
            v1 = vector_value(vector1)
            v2 = vector_value(vector2)
            v1_v2 = np.dot(vector1, vector2)
            return (v1_v2 / (v1 * v2)) * 100
        
        def generate_test(self):
            sentences = nlp.sent_tokenize(self.summary)
            cp = nlp.RegexpParser(self.grammar)
            question_answer_dict = dict()
            for sentence in sentences:
                tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
                tree = cp.parse(tagged_words)
                for subtree in tree.subtrees():
                    if subtree.label() == "CHUNK":
                        temp = ""
                        for sub in subtree:
                            temp += sub[0]
                            temp += " "
                        temp = temp.strip()
                        temp = temp.upper()
                        if temp not in question_answer_dict:
                            if len(nlp.word_tokenize(sentence)) > 20:
                                question_answer_dict[temp] = sentence
                        else:
                            question_answer_dict[temp] += sentence
            keyword_list = list(question_answer_dict.keys())
            question_answer = list()
            for i in range(int(self.noOfQues)):
                rand_num = np.random.randint(0, len(keyword_list))
                selected_key = keyword_list[rand_num]
                answer = question_answer_dict[selected_key]
                rand_num %= 4
                question = self.question_pattern[rand_num] + selected_key + "."
                question_answer.append({"Question": question, "Answer": answer})
            test = []
            for i, qa in enumerate(question_answer):
                test.append(f"{i+1}. {qa['Question']}\nAnswer: {qa['Answer']}\n")
            return test


    output = SubjectiveTest(data, noOfQues)
    test = output.generate_test()
    #print("".join(test))
 


    return render_template('qsandansr.html',out= "".join(test))



if __name__ == '__main__':
    app.run(debug = True,port =8000 )