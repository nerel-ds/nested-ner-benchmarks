import os
import os.path as osp
from dataclasses import dataclass
from typing import List

import pymorphy2
import tokenizations
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


COLL_TACRED_FNAMES = open("resources/coll_tacred_fnames.txt").read().split("\n")


@dataclass
class DataRow:
    token: str
    normal_form: str
    pos: str
    ner: List[str]
    str_ner: str


def get_biaffine_doc_dicts(txt_fps):
    doc_dicts = []
    ner_types = set()
    unmatched_entities_count = 0
    for txt_num, txt_fp in enumerate(txt_fps):  # enumerate([random.choice(txt_fps)]): #
        ann_fp = txt_fp.replace(".txt", ".ann")
        txt = open(txt_fp).read()
        ann = open(ann_fp).read()

        entities = []
        for row in ann.split("\n"):
            if row.startswith("T"):
                try:
                    raw_id, ent_start_end_pos, ent_text = row.split("\t")
                    ent_id = int(raw_id[1:])
                    ent_type, start, end = ent_start_end_pos.split()
                    start = int(start)
                    end = int(end)
                    entity = (ent_id, ent_type, start, end, ent_text)
                    entities.append(entity)
                except Exception as e:
                    print(f'Exception on row "{row}" with "{e}"')

        sent_start = sent_end = 0
        doc_dict = dict(
            doc_key=txt_fp[5:],
            ners=[],
            sentences=[],
        )
        selected_ent_ids = []
        for i, sent in enumerate(txt.split("\n")):
            sent_ents = []
            # print('Sentence length', len(sent))
            sent_start = sent_end
            sent_end += len(sent) + 1
            # if 'coll3_tacred' in txt_fp: sent_end += 1
            if os.path.basename(txt_fp) in COLL_TACRED_FNAMES:
                sent_end += 1
            sent_tokens = word_tokenize(sent, language="russian")
            sent_spans = tokenizations.get_original_spans(sent_tokens, sent)
            # print('Sentence bounds', sent_start, sent_end)
            for ent in entities:
                ent_id, ent_type, ent_start, ent_end, ent_text = ent
                ner_types.add(ent_type)
                ent_token_ids = []
                if ent_start >= sent_start and ent_end <= sent_end:
                    # print((ent_start, ent_end))
                    for token_id, span in enumerate(sent_spans):
                        if (
                            span
                            and span[0] >= (ent_start - sent_start)
                            and span[1] <= (ent_end - sent_start)
                        ):
                            ent_token_ids.append(token_id)
                if ent_token_ids:
                    # ent = (ent_id, ent_type, ent_start, ent_end, ent_text, ent_token_ids[0], ent_token_ids[-1])
                    ent = (ent_token_ids[0], ent_token_ids[-1], ent_type)
                    selected_ent_ids.append(ent_id)
                    sent_ents.append(ent)
            if sent_tokens:
                doc_dict["ners"].append(sent_ents)
                doc_dict["sentences"].append(sent_tokens)
            # if i == 3:
            #     break
        unmatched_entities = [
            ent for ent in entities if (ent[0] not in selected_ent_ids)
        ]  # and (ent[3] <= sent_end)
        if len(unmatched_entities) > 0:
            # txt_num_to_begin += txt_num + 1
            # print(txt_num, txt_fp)
            # print('Unmatched entities: ')
            # pprint(unmatched_entities)
            unmatched_entities_count += len(unmatched_entities)
            # break
        doc_dicts.append(doc_dict)

    print(
        "Total ents: ",
        sum(
            [len(sent_ents) for doc_dict in doc_dicts for sent_ents in doc_dict["ners"]]
        ),
    )
    print("Unmatched ents: ", unmatched_entities_count)
    print("NER types: ", ner_types)

    return doc_dicts


def prepare_pyramid_doc_dicts(doc_dicts):
    pyramid_doc_dicts = []
    for doc_dict in doc_dicts:
        for tokens, ners in zip(doc_dict["sentences"], doc_dict["ners"]):
            pyramid_doc_dicts.append(
                {
                    "tokens": tokens,
                    "entities": [
                        {"entity_type": t, "span": [s, e + 1]} for s, e, t in ners
                    ],
                }
            )
    return pyramid_doc_dicts


def prepare_pyramid_text_doc_dicts(doc_dicts):
    pyramid_doc_dicts = []
    for doc_dict in doc_dicts:
        shift = 0
        pyramid_doc_dict = {
            "tokens": [],
            "entities": [],
        }
        for tokens, ners in zip(doc_dict["sentences"], doc_dict["ners"]):
            pyramid_doc_dict["tokens"] += tokens
            pyramid_doc_dict["entities"] += [
                {"entity_type": t, "span": [s + shift, e + shift + 1]}
                for s, e, t in ners
            ]
            shift += len(tokens)
        pyramid_doc_dicts.append(pyramid_doc_dict)
    return pyramid_doc_dicts


def sort_entities(ents):
    """
    (1) entity mentions starting earlier have priority over entities
    starting later, 
    (2) for mentions with the same
    beginning, longer entity mentions have priority
    over shorter ones.
    """
    return sorted(ents, key=lambda x: (x[0], -(x[1] - x[0] + 1)))


def prepare_seq2seq_doc_dicts(doc_dicts):
    ma = pymorphy2.MorphAnalyzer()
    data_rows = []
    for dd in tqdm(doc_dicts):
        for sent, sent_ners in zip(dd["sentences"], dd["ners"]):
            sent_ners = sort_entities(sent_ners)
            sent_data_rows = []
            for token in sent:
                pymorphy_res = ma.parse(token)
                if pymorphy_res is not None and len(pymorphy_res) > 0:
                    normal_form = pymorphy_res[0].normal_form
                    # for special symbols like .,"'
                    pos = pymorphy_res[0].tag.POS if pymorphy_res[0].tag.POS else token
                else:
                    normal_form = token
                    pos = None
                sent_data_rows.append(
                    DataRow(
                        token=token,
                        normal_form=normal_form,
                        pos=pos,
                        ner=[],
                        str_ner="",
                    )
                )
            # BILOU NER
            for ne in sent_ners:
                start, end, ne_type = ne
                if start == end:
                    # then it's Unit NE token
                    sent_data_rows[start].ner.append("U-" + ne_type)
                else:
                    # then it's Begin-Inside-Last NE token
                    sent_data_rows[start].ner.append("B-" + ne_type)
                    for i in range(start + 1, end):
                        sent_data_rows[i].ner.append("I-" + ne_type)
                    sent_data_rows[end].ner.append("L-" + ne_type)
            for dr in sent_data_rows:
                # fill Outside tokens
                if len(dr.ner) == 0:
                    dr.ner.append("O")
                dr.str_ner = "|".join(dr.ner)
            data_rows.append(sent_data_rows)
    return data_rows


TAG2QUERY = {
    'AGE': 'Возраст - это период времени, когда кто-то был жив или что-то существовало.',
    'AWARD': 'Награда - это приз или денежная сумма, которую человек или организация получают за достижение.',
    'CHARGE': 'Обвинение - это официальное заявление полиции о том, что кто-то обвиняется в преступлении.',
    'CITY':'Город - это место, где живет много людей, со множеством домов, магазинов, предприятий и т.д.',
    'COUNTRY':'Страна - это территория земли, на которой есть собственное правительство, армия и т.д.',
    'CRIME': 'Преступление - это действие или деятельность, противоречащая закону, или незаконная деятельность в целом.',
    'DATE':'Дата - это номер дня в месяце, часто указываемый в сочетании с названием дня, месяца и года.',
    'DISEASE':'Болезнь - это заболевание людей, животных, растений и т.д., вызванное инфекцией или нарушением здоровья.',
    'DISTRICT':'Район - это территория страны или города, имеющая фиксированные границы, которые используются для официальных целей, или имеющая особую особенность, которая отличает его от окружающих территорий.',
    'EVENT': 'Событие - это все, что происходит, особенно что-то важное или необычное.',
    'FACILITY': 'Объект - это место, включая здания, где происходит определенная деятельность.',
    'FAMILY': 'Семья - это группа людей, связанных друг с другом, таких как мать, отец и их дети.',
    'IDEOLOGY': 'Идеология - это набор убеждений или принципов, особенно тех, на которых основана политическая система, партия или организация.',
    'LANGUAGE': 'Язык - это система общения, используемая людьми, живущими в определенной стране.',
    'LAW': 'Закон - это правило, обычно устанавливаемое правительством, которое используется для упорядочивания поведения общества.',
    'LOCATION': 'Местоположение - это место или позиция.',
    'MONEY': 'Деньги - это монеты или банкноты с указанием их стоимости, которые используются для покупки вещей, или их общая сумма.',
    'NATIONALITY': 'Национальность - это группа людей одной расы, религии, традиций и т.д.',
    'NUMBER': 'Число - это знак или символ, представляющий единицу, которая является частью системы подсчета и расчета.',
    'ORDINAL': 'Порядковый номер - это число, которое показывает положение чего-либо в списке.',
    'ORGANIZATION': 'Организация - это компания или другая группа людей, которые работают вместе для определенной цели.',
    # 'OUT': '[UNK]',
    'PERCENT': 'Процент - это одна часть из каждых 100 или указанное количество чего-либо, деленное на 100.',
    'PERSON': 'Человек - мужчина, женщина или ребенок.',
    'PENALTY': 'Штраф - это вид наказания, часто включающий выплату денег, который назначается вам, если вы нарушите соглашение или не соблюдаете правила.',
    'PRODUCT': 'Продукт - это то, что предназначено для продажи, особенно что-то произведенное в результате промышленного процесса или что-то выращенное в сельском хозяйстве.',
    'PROFESSION': 'Профессия - это любой вид работы, требующий специальной подготовки или определенных навыков.',
    'RELIGION': 'Религия - это вера и поклонение богу или богам или любая подобная система веры и поклонения.',
    'STATE_OR_PROVINCE': 'Штат или провинция - одна из областей, на которые страна или империя делятся в рамках организации своего правительства, которое часто имеет некоторый контроль над своими собственными законами.',
    'TIME': 'Время - это часть существования, которая измеряется минутами, днями, годами и т.д., или его процесс, рассматриваемый как единое целое.',
    'WORK_OF_ART': 'Произведение искусства - это предмет, созданный творцом большого мастерства, особенно картина, рисунок или статуя.',
}

def prepare_mrc_doc_dicts(doc_dicts):
    sample_id = 0

    new_doc_dicts = []
    for dd in doc_dicts:
        for sent, sent_ners in zip(dd["sentences"], dd["ners"]):
            label_id = 1 # in official implementations label counter starts from 1
            for tag, query in TAG2QUERY.items():
                tag_ners = [i for i in sent_ners if i[2] == tag]
                start_positions = [i[0] for i in tag_ners]
                end_positions = [i[1] for i in tag_ners]
                new_doc_dicts.append({
                    'context': ' '.join(sent),
                    'entity_label': tag,
                    'impossible': False,
                    'qas_id': f'{sample_id}.{label_id}',
                    'query': query,
                    'start_position': start_positions,
                    'end_position': end_positions,
                    'span_position': [f'{s};{e}' for (s, e) in zip(start_positions, end_positions)],
                })
                label_id += 1
            sample_id += 1
    return new_doc_dicts


