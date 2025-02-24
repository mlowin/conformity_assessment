from flask import Flask, render_template, send_from_directory, request, redirect
import ntpath, os
from flask_sqlalchemy import SQLAlchemy
import json
import sys
import numpy as np
sys.path.append("..")
#import flask_app.compute_metrics as compute_metrics

import getpass
username = getpass.getuser()
passwords = {
    'Maximilian Lowin': '',
    'ubuntu': 'DSS%40j4ilbr4€k', #DSS@j4ilbr4€k
    'maxim': ''
}
# create the extension
db = SQLAlchemy()
#gunicorn -w 4 -b 0.0.0.0 'server:app'
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:"+passwords[username]+"@localhost:3306/dss_jailbreak"
db.init_app(app)

class Usecase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), unique=True, nullable=False)
    weights = db.Column(db.String(10000))
    description = db.Column(db.String(1000))
    is_custom = db.Column(db.Boolean,nullable=False)
    rating = db.Column(db.Float, nullable=True)
    assessments = db.Column(db.Integer)


class Question(db.Model):
    id = db.Column(db.String(100), unique=True, nullable=False, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(1000))

class Answer(db.Model):
    id = db.Column(db.String(100), unique=True, nullable=False, primary_key=True)
    question = db.Column(db.String(100), primary_key=True)
    description = db.Column(db.String(1000))

with app.app_context():
    db.create_all()

@app.route("/")
def usecase_list():
    usecases = Usecase.query.filter_by(is_custom=True).all()
    return render_template('usecase_selection.htm',usecases=usecases)

@app.route("/usecase/database")
def usecase_database():
    usecases = Usecase.query.filter_by(is_custom=False).all()
    return render_template('usecase_database.htm',usecases=usecases)

@app.route("/usecase/create")
def usecase_create():
    data = {}
    with open('config/structure.json', 'r') as f:
        data = json.load(f)

    questions = Question.query.all()
    question_struct = {}
    for question in questions:
        if question.category not in question_struct:
            question_struct[question.category] = {}
        
        question_struct[question.category][question.id] = {
            'description': question.description,
            'active': False,
            'answers': {}
        }
        answer_objs = Answer.query.filter_by(question=question.id).all()
        for answer in answer_objs:
            question_struct[question.category][question.id]['answers'][answer.id] = {
                'value': answer.description
            }

    return render_template('usecase_create.htm', structure=data, questions=question_struct, uid=0)

@app.route("/usecase/copy/<id>", methods=['GET'])
def usecase_copy(id):
    usecase_db = Usecase.query.filter_by(id=id).first()    
    usecase_new = Usecase(title=usecase_db.title+" (custom)", description="Please modify for your scenario", weights=usecase_db.weights, is_custom=True, assessments=0)
    db.session.add(usecase_new)
    db.session.commit()
    return redirect("/")

@app.route("/usecase/delete/<id>", methods=['GET'])
def usecase_delete(id):
    usecase_db = Usecase.query.filter_by(id=id).delete()    
    db.session.commit()
    print("DELETE",id)
    return redirect("/")

@app.route("/usecase/<id>", methods=['GET'])
def usecase_edit(id):
    usecase = Usecase.query.filter_by(id=id).first()
    # Metrics
    structure = json.loads(usecase.weights)
    ## Fill Metrics from Database with structure.js
    with open('config/structure.json', 'r') as f:
        structure_config = json.load(f)
        for item_id, item in structure_config.items():
            for sub_item_id, sub_item in item['children'].items():
                if sub_item_id not in structure[item_id]['children']:
                    structure[item_id]['children'][sub_item_id] = {
                        'title': sub_item['title'],
                        'description': sub_item['description'],
                        'rules': {                            
                            'active': False,
                            'green_min': '',
                            'green_max': '',
                            'yellow_min': '',
                            'yellow_max': '',
                        }
                    }
    # Questions
    questions = Question.query.all()
    question_struct = {}
    for question in questions:
        if question.category not in question_struct:
            question_struct[question.category] = {}
        
        active = question.id in structure['questions']
        multi = False
        if active:
            multi = structure['questions'][question.id]['multi']
        answer_objs = Answer.query.filter_by(question=question.id).all()
        answers = []
        for answer in answer_objs:
            rank = ''
            if question.id in structure['questions'] and answer.id in structure['questions'][question.id]['children']:
                rank = structure['questions'][question.id]['children'][answer.id]
            answers.append({
                'id':answer.id,
                'description':answer.description,
                'rank':rank
            })
        question_struct[question.category][question.id] = {'description':question.description,'active':active,'multi':multi,'answers':answers}

    return render_template('usecase_create.htm', title=usecase.title, description=usecase.description, structure=structure, uid=int(id), questions=question_struct)

def generate_structure(data):
    structure = {}
    print("gen")
    #structure based on explainability, performance & Fairness
    with open('config/structure.json', 'r') as f:
        print("open2")
        structure = json.load(f)
        print("open")
        for item_key in structure:
            structure[item_key]['description'] = data['description_'+item_key]
            for sub_key in structure[item_key]['children']:
                rules = {
                    'active': 'active_'+sub_key in data,
                    'green_min': 0 if 'green_min_'+sub_key not in data else data['green_min_'+sub_key],
                    'green_max': 0 if 'green_max_'+sub_key not in data else data['green_max_'+sub_key],
                    'yellow_min': 0 if 'yellow_min_'+sub_key not in data else data['yellow_min_'+sub_key],
                    'yellow_max': 0 if 'yellow_max_'+sub_key not in data else data['yellow_max_'+sub_key]
                }
                structure[item_key]['children'][sub_key]['rules'] = rules
    # structure based on questions
    questions = {}
    print("genz")
    
    for item_q in data:
        if item_q[:8] == 'question':
            parts = item_q.split('-')
            question = parts[1]
            question_obj = {
                'multi': 'multi-'+question in data,
                'children': {}
            }
            for item_a in data:
                if item_a[:6] == 'answer':
                    parts = item_a.split('-')
                    if parts[1] == question and data[item_a] != 'inactive':
                        question_obj['children'][parts[2]] = data[item_a]
            questions[question] = question_obj
            print(question_obj)
        structure['questions'] = questions
    return structure
@app.route("/usecase", methods=['POST'])
def usecase_store():
    data = request.form
    structure = generate_structure(data)    
    weights = json.dumps(structure)
    usecase = Usecase(title=data['title'], description=data['description'], weights=weights)
    db.session.add(usecase)
    db.session.commit()
    return redirect("/")
    
@app.route("/usecase/<id>", methods=['POST'])
def usecase_update(id):
    data = request.form
    structure = generate_structure(data)    
    weights = json.dumps(structure)
    
    print("?!?")
    usecase = Usecase.query.filter_by(id=id).first()
    usecase.title = data['title']
    usecase.description = data['description']
    print("fs?!?")
    usecase.weights = weights
    db.session.commit()
    print("??")
    return redirect("/")

@app.route("/get_columns", methods=['POST'])
def get_files():
    data = request.form
    import pandas as pd
    df = pd.read_csv('../datasets/'+data['file'])
    cols = list(df.columns)
    return json.dumps(cols)
    

@app.route("/evaluation/files/<id>")
def evaluation(id):
    from os import listdir
    from os.path import isfile, join 
    files_dataset = [{'filename':f,'disabled':f != 'Fraud_Germany_2023.csv'} for f in listdir('../datasets') if isfile(join('../datasets', f))]
    files_model = [{'filename':f,'disabled':f != 'Fraud_Classifier_XGBoost.sav'} for f in listdir('../models') if isfile(join('../models', f))]
    return render_template('evaluation_select_files.htm',files_dataset=files_dataset, files_model = files_model, id=id)


@app.route("/evaluation/questions/<id>",methods=['GET','POST'])
def evaluation_questions(id):
    usecase = Usecase.query.filter_by(id=id).first()
    structure = json.loads(usecase.weights)
    questions = {}
    for question_id, question_elem in structure['questions'].items():
        question_obj = Question.query.filter_by(id=question_id).first()
        question = {'description': question_obj.description, 'answers':{}, 'multi': question_elem['multi'] if 'multi' in question_elem else False}
        for answer_id in question_elem['children']:
            answer_obj = Answer.query.filter_by(id=answer_id).first()
            question['answers'][answer_id] = answer_obj.description
        if question_obj.category not in questions:
            questions[question_obj.category] = {}
        questions[question_obj.category][question_id] = question
    
    data = request.form
    outcome_col = data['outcome']
    sensitive_attr = data['sensitive']
    model = data['model']
    dataset = data['dataset']
    return render_template('evaluation_questions.htm',questions=questions, id=id, outcome=outcome_col, sensitive=sensitive_attr, model = model, dataset=dataset)



@app.route("/evaluation/report", methods=['POST'])
def evaluation_report():
    data = request.form

    #load dataframe and 
    import pandas as pd
    
    import pickle

    read_cache = True

    if read_cache:
        file = open('metrics.pickle', 'rb')
        metrics = pickle.load(file)
        file.close()
    # else:
    #     # df = pd.read_csv('../datasets/'+data['dataset'])
    #     # outcome_col = data['outcome']
    #     # sensitive_attr = data['sensitive']
    #     # model_path = '../models/' + data['model']
    #     # metrics = compute_metrics.compute_metrics(Data=df, model_name=model_path,
    #     #                                         column=outcome_col, sensitive_attr=sensitive_attr)
    #     file = open('metrics.pickle', 'wb')
    #     pickle.dump(metrics, file)
    #     file.close()
    #     file = open('metrics.pickle', 'wb')
    #     pickle.dump(metrics, file)
    #     file.close()
        
    f = open('config/structure.json', 'r')
    structure_json = json.load(f)

    #load selected use case
    usecase = Usecase.query.filter_by(id=data['usecase']).first()
    structure = json.loads(usecase.weights)

    # plotly data
    dct_history = {
        'fairness_stat_par_mean': [0.2, 0.3, 0.1],
        'fairness_eq_odds_mean': [0.21, 0.10,0.17],
        'fairness_eq_opp_mean': [0.4, 0.5, 0.6],
        'explainability_mean_stability': [0.01, 0.02, 0.03],
        'performance_accuracy_mean': [0.41, 0.51, 0.63],
        'performance_balanced_acc_mean': [0.30, 0.44, 0.66],
        'performance_precision_mean': [0.7, 0.73, 0.81],
        'performance_recall_mean': [0.9, 0.87, 0.71],
        'performance_f1_mean': [0.6, 0.66, 0.71],
        'performance_data_drift': [0.0, 0.01, 0.0],
        'explainability_dataset_documentation':['documentation_z_no', 'documentation_z_no', 'documentation_preprocessing'],
        'explainability_dataset_preprocessing': ['data_pre_no', 'data_pre_hypothesis', 'data_pre_hypothesis'],
        'explainability_dataset_subpopulation': ['no_sub', 'yes_sub', 'yes_sub'],
        'explainability_model_algorithm': ['no_algo', 'no_algo', 'yes_exp_performance'],
        'explainability_model_prediction': ['mp_confidence', 'mp_confidence', 'mp_confidence'],
        'explainability_model_hyper': ['no_hyper', 'no_hyper', 'yes_hyper'],
        'explainability_model_intended_use': ['yes_use', 'yes_use', 'yes_use'],
        'explainability_model_intended_user': ['yes_user', 'yes_user', 'yes_user'],
        'explainability_output_explaination': ['no_explaination', 'yes_counterfactural', 'yes_global'],
        'explainability_output_metric': ['yes_metric', 'yes_metric', 'yes_metric'],
        'explainability_preprocessing_cleaning': ['no_preprocessing', 'yes_preprocess', 'yes_cleaned'],
    }
    from datetime import date 

    today = date.today()
    lst_labels = ['2023-04-03', '2023-09-12', '2024-04-30', today]
    dct_plots = {}

    #check traffic light for each (sub)item
    items_to_delete = []
    descriptions = {}

    all_thresholds = structure#.copy()
    for item_id, item in structure.items():
        if item_id == 'questions':
            continue
        item_color = 'green'
        for sub_id, sub in item['children'].items():
            if not sub['rules']['active']:
                items_to_delete.append((item_id,sub_id))
            else:
                if 'requires' in structure_json[item_id]['children'][sub_id]:
                    requirement_fulfilled = True
                    for req in structure_json[item_id]['children'][sub_id]['requires']:
                        if not (('answer-'+req['question'] in data and data['answer-'+req['question']] == req['answer']) or ('answer-'+req['question']+'-'+req['answer'] in data)):
                            requirement_fulfilled = False
                            break
                    if not requirement_fulfilled:
                        items_to_delete.append((item_id,sub_id))
                        continue
                
                value = metrics[item_id][sub_id]
                rules = sub['rules']
                color = get_color(value, rules)
                sub['color'] = color
                sub_info = get_sub_info(item_id, sub_id)
                if sub_info['description'] is not None and sub_info['description'] != '':
                    sub['description'] = sub_info['description']
                else:
                    sub['description'] = "None"
                sub['value'] = round(value,3) if (type(value) == float or type(value) == np.float64) else value
                sub['description'] = sub['description'].replace('XXX', str(round(sub['value']*100,1)))
                sub['tooltip'] = generate_tooltip(rules)
                if item_color == 'green' and color != 'green':
                    item_color = color
                elif item_color == 'yellow' and color == 'red':
                    item_color = color
                plotly_id = item_id+"_"+sub_id
                if plotly_id in dct_history:
                    history = dct_history[plotly_id]
                    history.append(value)
                    colors = get_discrete_colors(history, rules)
                    plotly_json =  build_plotly_json(history, lst_labels, colors)
                    dct_plots[plotly_id] = plotly_json
        item['color'] = item_color

    for item_sub in items_to_delete:
        del structure[item_sub[0]]['children'][item_sub[1]]
    questions = structure['questions']
    for q_id, question in questions.items():
        a_ids = []
        if question['multi']:
            for item in data:
                if 'answer-'+q_id+'-' in item:
                    a_ids.append(data[item])
        else:
            a_ids.append(data['answer-'+q_id])
        answer_titles = []
        color = 'green'
        for a_id in a_ids:
            answer_obj = Answer.query.filter_by(id=a_id).first()
            answer_titles.append(answer_obj.description)
            a_color = structure['questions'][q_id]['children'][a_id]
            if a_color == 'red' or (color == 'green' and a_color == 'yellow'):
                color = a_color
        answer_title = ', \r\n<br/>'.join(answer_titles)
        question_obj = Question.query.filter_by(id=q_id).first()
        question_title = question_obj.description
        question_category = question_obj.category
        css_category = question_category.replace(' ', '')
        if question_category not in structure['explainability']['children']:
            structure['explainability']['children'][question_category] = {
                'color': 'green',
                'children': {},
                'title': question_category,
                'value': 'question',
                'css_category': css_category
            }
        structure['explainability']['children'][question_category]['children'][q_id] = {
            'title': question_title,
            'value': answer_title,
            'color': color,
            'css_category': css_category
        }
        structure['explainability']['children'][question_category]['color'] = get_color_min(structure['explainability']['children'][question_category]['color'], color)
        plotly_id = "explainability_"+q_id
        if False and plotly_id in dct_history:
            history = dct_history[plotly_id]
            colors = []
            titles = []
            for his_item in history:
                colors.append(structure['questions'][q_id]['children'][his_item])                
                answer_obj = Answer.query.filter_by(id=his_item).first()
                titles.append(answer_obj.description)
            titles.append(answer_title)
            colors.append(color)
            plotly_json =  build_plotly_question_json(titles, lst_labels, colors)
            dct_plots[plotly_id] = plotly_json
        if color == 'yellow' and structure['explainability']['color'] == 'green':
            structure['explainability']['color'] = 'yellow'
        elif color == 'red' and structure['explainability']['color'] != 'red':
            structure['explainability']['color'] = 'red'

    del structure['questions']
    #load item descriptions for modals (UI stuff)
    modals = []
    for item_id, item in structure.items():
        modals.append({
            'id': item_id,
            'title': item['title'],
            'text': item['description']
        })

    for item_id, item in descriptions.items():
        modals.append({
            'id': 'child_'+item_id,
            'title': item['title'],
            'text': item['description']
        })
    
    file = open('metrics_precompute.json', 'r')
    all_metrics = file.read()
    all_thresholds = json.dumps(structure)
    file.close()
    from datetime import datetime

    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M:%S")

    return render_template('evaluation.htm',items=structure, modals=modals, title=usecase.title, questions=questions, plots=dct_plots, all_metrics=all_metrics, all_thresholds=all_thresholds,date=date)

def get_color_min(color1, color2):
    if color1 == 'green' and color2 == 'green':
        return 'green'
    if color1 == 'red' or color2 == 'red':
        return 'red'
    return 'yellow'

def generate_tooltip(rules):
    if rules['yellow_min'] > rules['green_min']: #Wenn kleinere Werte bei der Metrik besser sind
        return 'Decision-Rule: value ≤ '+rules['green_max']+": green; values ≤ "+rules['yellow_max']+": yellow; red otherwise"
    else:
        return 'Decision-Rule: value < '+rules['yellow_min']+": red; values < "+rules['yellow_max']+": yellow; green otherwise"

def build_plotly_json(history, labels, colors):
    import plotly
    import plotly.io as pio

    pio.templates.default = "plotly_white"
    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame({
      'labels': labels,
      'history': history,
    })

    fig = px.scatter(
        df, 
        x='labels', 
        y='history', 
        color=colors,
        color_discrete_map={'red':'#c34639', 'green':'#39cc77','yellow':'#e9f128'}
    )
    fig.update_layout(
        yaxis=dict(
            range=[0, 1]
        ),
        autosize=False,
        width=500,
        height=200,        
        showlegend=False,
        
    )
    fig.update(
        layout_coloraxis_showscale=False
    )
    
    fig.update_traces(marker=dict(size=12))
    fig.update_xaxes(title='')
    fig.update_yaxes(title='')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def build_plotly_question_json(titles, labels, colors):
    import plotly
    import plotly.io as pio

    pio.templates.default = "plotly_white"
    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame({
      'labels': labels,
      'history': titles,
      'values': [.5 for i in range(len(labels))]
    })

    fig = px.scatter(
        df, 
        x='labels', 
        y='values',
        hover_name='history',
        color=colors,
        color_discrete_map={'red':'#c34639', 'green':'#39cc77','yellow':'#e9f128'}
    )
    fig.update_layout(
        yaxis=dict(
            range=[0, 1]
        ),
        autosize=False,
        width=500,
        height=200,        
        showlegend=False,
        
    )
    fig.update_yaxes(visible=False)
    fig.update(
        layout_coloraxis_showscale=False
    )
    
    fig.update_traces(marker=dict(size=12))
    fig.update_xaxes(title='')
    fig.update_yaxes(title='')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def get_sub_info(item_id, child_id):    
    description = None
    with open('config/structure.json', 'r') as f:
        data = json.load(f)
        elem = data[item_id]['children'][child_id]
        if 'description' in elem:
            description = elem['description']
        return {
            'description': description,
            'title': elem['title']
        }

def get_color_rules(rules):
    if rules['yellow_min'] > rules['green_min']: #Wenn kleinere Werte bei der Metrik besser sind
        return [
            [float(rules['green_min']), '#39cc77'], [float(rules['green_max']), '#39cc77'], 
            [float(rules['yellow_min']), '#e9f128'], [float(rules['yellow_max']), '#e9f128'], 
            [float(rules['yellow_max']), '#c34639'], [1, '#c34639']
        ]
    else: 
        return [
            [0, '#c34639'], [float(rules['yellow_min']), '#c34639'], 
            [float(rules['yellow_min']), '#e9f128'], [float(rules['yellow_max']), '#e9f128'], 
            [float(rules['green_min']), '#39cc77'], [float(rules['green_max']), '#39cc77']
        ]
 

def get_color(value, rules):
    #if rules['yellow_min'] > rules['green_min']: #Wenn kleinere Werte bei der Metrik besser sind
    if value >= float(rules['green_min']) and value <= float(rules['green_max']):
        return 'green'
    if value >= float(rules['yellow_min']) and value <= float(rules['yellow_max']):
        return 'yellow'
    return 'red'

def get_discrete_colors(history, rules):
    colors = []
    for item in history:
        colors.append(get_color(item, rules))
    return colors


@app.route("/question", methods=['GET'])
def all_questions():
    questions = Question.query.all()
    return render_template('question_overview.htm',questions=questions)

@app.route("/question/new")
def question():
    return render_template('question_create.htm')

@app.route("/question/<id>")
def question_edit(id):
    question = Question.query.filter_by(id=id).first()
    answers = Answer.query.filter_by(question=id).all()
    return render_template('question_create.htm',id=id,old_id = id, description=question.description, category=question.category,answers=answers)

@app.route("/question", methods=['POST'])
def question_store():
    data = request.form
    if data['old_id'] != 0:
        question = Question.query.filter_by(id=data['old_id']).first()
        if question is not None:
            db.session.delete(question)
        answers = Answer.query.filter_by(question=data['old_id']).all()
        for answer in answers:
            db.session.delete(answer)
    question = Question(id=data['id'], description=data['description'], category=data['category'])
    answer_ids = data.getlist('answer_id[]')
    key = 0
    for answer in data.getlist('answer[]'):
        answer = Answer(id=answer_ids[key], description=answer, question=data['id'])
        db.session.add(answer)
        key += 1
    db.session.add(question)
    db.session.commit()
    return redirect("/question")
    
#Sämtlicher Pfad, welcher vorher nicht aufgelistet wurde, wird nun hier bearbeitet (bspw. die normalen Seiten)
@app.route('/<path:path>', methods=['GET'])
def dynamic_site(path):
    #falls asset im Seitenordner existiert, gib diesen zurück
    filename = ntpath.basename(path)
    if filename != '':

        if os.path.isfile('assets/'+path):
            parts = path.split("/")
            raw_path = "/".join(parts[:-1])
            return send_from_directory('assets/'+raw_path, filename)
         
    return "-1"
   
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000,use_reloader=False)