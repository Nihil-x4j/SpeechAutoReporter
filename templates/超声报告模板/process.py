import os
import time
import re
import json
from tqdm import tqdm
from typing import Dict
'''
该脚本功能：
1.过滤"超声报告模板.txt"中存在的内容错误情况。
2.保存成多个可以验证和内容生成的正则表达式模板。
'''

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = '超声报告模板.txt'
output_name = '超声报告模板_new.jsonl'
input_path = os.path.join(current_dir, file_name)
output_path = os.path.join(current_dir, output_name)

import dashscope
from dashscope import Generation

def filter_content(file_path):
    """
    内容过滤函数:
    1. 过滤掉空白行和行首尾空白符。
    2. 过滤掉存在（第*章）（第*节）所在行
    输入: 文件路径
    返回: 过滤后的行列表
    """
    filtered_lines = []
    with open(file_path, 'r', encoding='GBK') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if '第' in stripped and ('章' in stripped or '节' in stripped):
                continue
            filtered_lines.append(stripped)
    return filtered_lines

def split_diseases(lines):
    """
    疾病分割函数，假设文档list中【】包裹的疾病名称，从【】开始到下一个【】之间是疾病的报告模板，
    输入是字符串列表，输出是字典，键是疾病名称，值是疾病字符串。
    """
    disease_dict = {}
    current_disease = None
    buffer = []
    for line in lines:
        if line.startswith('【') and line.endswith('】'):
            if current_disease and buffer:
                disease_dict[current_disease] = '\n'.join(buffer).strip()
            current_disease = line[1:-1]
            buffer = [f'疾病：{current_disease},模板内容：']
        else:
            if current_disease:
                buffer.append(line)
    if current_disease and buffer:
        disease_dict[current_disease] = '\n'.join(buffer).strip()
    return disease_dict

def process_and_save_to_jsonl(input_path, output_jsonl_path):
    """
    串联过滤和分割函数，并将结果保存为 JSONL 格式文件。
    """
    filtered_lines = filter_content(input_path)
    disease_dict = split_diseases(filtered_lines)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for disease_name, template_text in disease_dict.items():
            line = json.dumps({disease_name: template_text}, ensure_ascii=False)
            f.write(line + '\n')

    return disease_dict

def call_dashscope_model(prompt: str, model_name: str = "qwen-plus-latest", retries: int = 3, retry_delay: int = 10) -> str:
    """
    使用 DashScope 平台调用大模型 API，输入提示词，返回模型输出结果，并处理网络异常重试。
    """
    dashscope.api_key = "sk-7ba7379293084b40bd1d50d03fa71af5"
    attempt = 0
    while attempt < retries:
        try:
            #print(prompt)
            response = Generation.call(
                model=model_name,
                messages = [
                    {'role': 'user', 'content': f'{prompt}'}
                ],
                parameters={
                    "temperature": 0.3,
                    "enable_thinking": True
                },
                extra_body={"enable_thinking": True,
                            "thinking_budget": 3000}
            )
            print(response)
            #output_text = response['output']["choices"][0]["message"]["content"].strip()
            output_text = response['output']["text"].strip()
            return output_text
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise RuntimeError("达到最大重试次数，仍然无法完成请求。")
            else:
                print(f"尝试 {attempt}/{retries} 失败，等待 {retry_delay} 秒后重试...")
                print(f"{str(e)}")
                time.sleep(retry_delay)

def call_regex_generation(prompt: str, max_retries=3):
    """
    调用大模型生成正则表达式，带重试机制。
    """
    out = None
    error_msg = []
    frist_prompt = f'''
        你是一个医学模板文本生成助手。请根据用户提供用OCR工具从pdf中识别的超声影像部门的结构化病例模板，自动生成一个对应的正则表达式（regex pattern）模板，用于校验病例是否符合该结构和字段要求。
        由于文本来自ocr识别，当且仅当文本存在明显错误时你可以自行纠正。
        可能会存在不合适的空格、换行或异常排版，模板可以不出现这些匹配项或者模糊匹配
        生成要求如下：
        1. 识别并提取固定文本部分，保持原样，其属于模板中标准化固定话术。
        2. 对文本中可能的可选字段进行识别和处理。
            (1) 模板中()、/等字符都会暗示其字段可能是可选项，一般/存在部分一定存在可选项，例如：
                - (左/右) → (左|右)
                - 清晰/不清晰 → (清晰|不清晰)
                - A/B/C → (A|B|C)
                大部分此类字段属于关键词必须存在，不能生成(?:清晰|不清晰)，因为这样会导致即使病例样本留空也会匹配成功会导致校验错误。
            (2) 仍然可能会存在一些较为复杂和特殊的情况需要重点关注，你必需结合上下文语义灵活处理，不能定式思维，例如：
            - ...和(或)... → ...(和|或)... (解释：模板这里约束必须填‘xxx和xxx’或者‘xxx或xxx’)
            - ...(右/左/双眼)前部/赤道部/后极部玻璃体... → ...(右|左|双眼)(前部|赤道部|后极部)玻璃体... (解释：模板这里存在连续的可选项“眼部+眼球方位”)
            - ...，(部分 病例可见)条带状弱回声中央局限增强呈类椭圆形。... → ...(?:，可见条带状弱回声中央局限增强呈类椭圆形)?。... (解释：模板指的是，这个现象可能在某些病例中出现，某些不出现)
            - ...良/恶性(可能性大)... → ...(良|恶)性(?:可能性大)... (解释：模板此处内容可以判断是确诊还是极大可能性确诊)
            - 声影(+), xxxx(-) → 声影\(+\), xxxx\(-\) (解释：不做处理，这里表示确认是阳性或阳性，但是需要保留括号)
            - 声影(+/-) → 声影\(+|-\) (解释：可能阴性可能阳性，医学习惯，这个检测项目需要保留括号)
            注意，阳性阴性检测结果都需要保留括号。
            (3) 还可能会存在一些多选情况需要判断
            - 左/右侧颈部(I 、Ⅱ 、Ⅲ 、IV 、V 、VI 、V)区域 → (左|右)侧颈部$([IVX]+)(?:[、,][IVX]+)*$
            (4) 可能存在很复杂和模糊的组合情况
            - (少量/大量)弱点状和(或)条状回声 → (?:少量|大量)(?:弱点状|条状|弱点状和条状)回声 (解释：如果是和表示都两种回声都出现，或只出现一个)
             - 与部位(如周边部、赤道部、后极部、黄斑区、视盘)紧密 相连或不与球壁回声相连 → 与((?:周边部|赤道部|后极部|黄斑区|视盘)紧密相连|不与球壁回声相连)(解释：模板此处相连和不相连互斥，所以是嵌套结构)
            总之，需要你从专业医生撰写报告的角度进行判断
        3. 对数值字段添加合理正则约束，例如：
        - 大小字段：cm x cm -> \s*\d+(\.\d+)?\s*cm×\s*\d+(\.\d+)?\s*cm
        - 数字范围：  ～ ->  \s*\d+(\.\d+)?\s*～\s*\d+(\.\d+)?
        - 占位符字段：\s*\d+\s*次
        一般而言，存在单位字段，不论模板前是否有空格，都应该存在可填项(整数或者实数)
        4. 输出结果应为raw string（r"..."）风格，不要包含其他解释说明，需要一再强调的是，输出的是raw字符串表示的正则表达式，1.左右括号的匹配请使用\(\),$不合法，禁止使用，2.对于+需要\+进行匹配。
        5. 模板中可能存在识别错误的字符或者错误的空白符等，请结合你的专业知识自行纠正，对于空白符应该较为宽松，不能要求严格匹配，你翻译的正则串只需要保证内容正确性，结构正确性不做考虑。
        6. 输入的疾病仅作为额外信息，不是模板内容本身，重点！注意返回字符串一定能编译
        
        示例1：
        <用户输入>
        疾病：脉络膜脱离,模板内容：\n超声所见：\n二维超声：(右/左/双)眼轴位切面探查360°全周或 部分玻璃体内可探及带状或凸向玻璃体内的弧形中强带状 回声，一端与周边部相连，另一端与赤道部或后极部球壁相 连，但不与视盘回声相连，类冠状切面检查可探及类花瓣状 弧形带状中强回声，运动试验(一),其下方为无回声区。\nCDFI: 玻璃体内带状回声上可探及丰富的血流信号， 不与视网膜中央动脉相延续，脉冲多普勒为与睫状后动脉  相同的动脉型血流频谱。\n超声提示：(右/左/双)眼玻璃体内异常回声，结合临 床考虑脉络膜脱离可能性大。
        <正确回答>
        r'超声所见：\s*二维超声：(右|左|双)眼轴位切面探查360°全周或部分玻璃体内可探及带状或凸向玻璃体内的弧形中强带状回声，一端与周边部相连，另一端与赤道部或后极部球壁相连，但不与视盘回声相连，类冠状切面检查可探及类花瓣状弧形带状中强回声，运动试验一，其下方为无回声区。\s*CDFI:玻璃体内带状回声上可探及丰富的血流信号，不与视网膜中央动脉相延续，脉冲多普勒为与睫状后动脉相同的动脉型血流频谱。\s*超声提示：(右|左|双)眼玻璃体内异常回声，结合临床考虑脉络膜脱离可能性大。'
        示例2：
        <用户输入>
        疾病：部分型肺静脉异位引流,模板内容：\n各项测值同上。\n1. 心房正位，心室右袢，左(右)位主动脉弓。\n2. 右心房、右心室明显增大，左心正常。\n3. 左心房壁上可见1～3条肺静脉回流入左心房；右\n心房壁有时可见肺静脉开口。\n4. 四腔心切面见房间隔中部连续性中断      mm ( 卵 圆孔未闭)。\n5. 肺动脉主干及其分支增宽。\n6.CDFI:    舒张期房水平可见过隔分流束。引流口处可 录及肺静脉血流频谱。\n7.  组织多普勒：二尖瓣环水平室间隔基底段Em>Am 或  Em<Am 。Vs:          cm/s,Va:         cm/s,Ve:         cm/s,Ve/Va>1。\n超声提示：先天性心脏病，部分型肺静脉异位引流，合 并房间隔缺损。\n*当发现右心房、右心室增大程度与房间隔缺损不成 正比时，要特殊观察四条肺静脉是否全部回流入左心房。 本病容易漏诊。"
        <正确回答>
        r'各项测值同上。\s*1\.心房正位，心室右袢，(左|右)位主动脉弓。\s*2\.右心房、右心室明显增大，左心正常。\s*3\.左心房壁上可见[1-3]条肺静脉回流入左心房；右心房壁有时可见肺静脉开口。\s*4\.四腔心切面见房间隔中部连续性中断\s*\d+(?:\.\d+)?\s*mm(卵圆孔未闭)\s*5\.肺动脉主干及其分支增宽。\s*6\.CDFI:舒张期房水平可见过隔分流束。引流口处可录及肺静脉血流频谱。\s*7\.组织多普勒：二尖瓣环水平室间隔基底段Em>Am或Em<Am。Vs:\s*\d+(.\d+)?\s*cm/s,Va:\s*\d+(.\d+)?\s*cm/s,Ve:\s*\d+(.\d+)?\s*cm/s,Ve/Va>1。\s*超声提示：先天性心脏病，部分型肺静脉异位引流，合并房间隔缺损。\s*当发现右心房、右心室增大程度与房间隔缺损不成正比时，要特殊观察四条肺静脉是否全部回流入左心房。\s*本病容易漏诊。'
        <用户输入>
        疾病：马方综合征,模板内容：\n各项测值同上。\n1. 左心房、左心室扩大，右心室、右心房正常。\n2. 主动脉根部内径增宽，升主动脉呈瘤样扩张，最宽处 内径     mm, 胸骨上窝探查主动脉弓内径恢复正常     mm, 降主动脉起始段内径     mm。\n3. 主动脉瓣无增厚，开放正常，关闭明显不严。或主 动脉瓣右、无冠瓣发育较长，舒张期体部脱向左心室流出 道侧，致瓣口关闭不严。\n4. 余瓣膜解剖形态及运动未见异常。\n5. 房、室间隔连续完整。\n6. 室间隔及左心室后壁厚度正常，振幅增强。\n7.CDFI:  舒张期左心室流出道内见源于主动脉瓣口 的红色花彩血流，面积      cm2, 流速(V)      m/s, 压 差 (PG)         mmHg。收缩期左心房内见源于二尖瓣口的蓝色 花彩血流，面积     cm2, 流速(V)          m/s, 压 差(PG)         mmHg。\n8.  组织多普勒：二尖瓣环水平室间隔基底段Em>Am 或\nEm<Am 。Vs:              cm/s,Va:          cm/s,Ve:         cm/s,Ve/Va>1。\n超声提示：马方综合征，或合并主动脉瓣脱垂，主动脉 瓣反流(轻、中、重度),二尖瓣轻度反流(相对性)。
        <正确回答>
        r'各项测值同上。\s*1\.左心房、左心室扩大，右心室、右心房正常。\s*2\.主动脉根部内径增宽，升主动脉呈瘤样扩张，最宽处内径\s*\d+(\.\d+)?\s*mm,胸骨上窝探查主动脉弓内径恢复正常\s*\d+(\.\d+)?\s*mm,降主动脉起始段内径\s*\d+(\.\d+)?\s*mm。\s*3\.主动脉瓣无增厚，开放正常，关闭明显不严。或主动脉瓣右、无冠瓣发育较长，舒张期体部脱向左心室流出道侧，致瓣口关闭不严。\s*4\.余瓣膜解剖形态及运动未见异常。\s*5\.房、室间隔连续完整。\s*6\.室间隔及左心室后壁厚度正常，振幅增强。\s*7\.CDFI:舒张期左心室流出道内见源于主动脉瓣口的红色花彩血流，面积\s*\d+(\.\d+)?\s*cm2,\s*流速$V$\s*\d+(\.\d+)?\s*m/s,\s*压差$PG$\s*\d+(\.\d+)?\s*mmHg。收缩期左心房内见源于二尖瓣口的蓝色花彩血流，面积\s*\d+(\.\d+)?\s*cm2,\s*流速$V$\s*\d+(\.\d+)?\s*m/s,\s*压差$PG$\s*\d+(\.\d+)?\s*mmHg。\s*8\.组织多普勒：二尖瓣环水平室间隔基底段Em>Am或Em<Am。Vs:\s*\d+(\.\d+)?\s*cm/s,Va:\s*\d+(\.\d+)?\s*cm/s,Ve:\s*\d+(\.\d+)?\s*cm/s,Ve/Va>1。\s*超声提示：马方综合征或合并主动脉瓣脱垂，主动脉瓣反流(轻|中|重)度,二尖瓣轻度反流\(相对性\)。'
        请根据以下输入文本，生成对应的正则表达式：
        {prompt}
    '''
    
    for i in range(max_retries):
        full_prompt = frist_prompt
        
        if len(error_msg) > 0:
            print(f"[尝试 {i}/{max_retries}]次：{error_msg}")
            full_prompt += f"上一次生成不符合要求，请检查内容并重新生成，上一次生成:{out}，错误信息{error_msg}"
            error_msg = []
        
        response = call_dashscope_model(full_prompt)
        out = f"{response}"

        if re.match(r'^r".*"$', response) or re.match(r"^r'.*'$", response):
            response = response[2:-1]
        else:
            error_msg.append(f"格式错误：返回不是raw（r''）字符串格式") 
        
        # if re.search(r'(?!(?<=\d)/(?=\d))/', response):
        #     error_msg.append(f"内容或格式错误：存在错误的/字符")
        
        if re.search(r'\$', response):
            error_msg.append(f"格式错误：存在禁止输出的$符")
        
        response = response.replace('\\\\','\\')
        
        try:
            re.compile(response, flags=re.MULTILINE | re.UNICODE)
        except re.error as e:
            error_msg.append(f"正则表达式无效：{str(e)}")
        
        if len(error_msg) == 0:
            return response
    raise ValueError(f"{error_msg}")

def generate_regex_jsonl(
    disease_templates_path: str,
    output_jsonl_path: str,
):
    """
    从 JSONL 文件中读取疾病模板字典，批量生成正则表达式，输出为 JSONL 文件。
    支持断点续传、失败跳过。
    """
    failed = []
    # Step 1: 从 JSONL 文件加载疾病模板
    disease_templates = {}
    with open(disease_templates_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            if isinstance(record, dict) and len(record) == 1:
                disease_name, template_text = next(iter(record.items()))
                disease_templates[disease_name] = template_text
            else:
                print(f"警告：无效的记录 {line.strip()}，跳过。")

    # Step 2: 加载已处理的疾病列表（用于断点续传）
    processed_diseases = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                processed_diseases.update(data.keys())

    # Step 3: 准备写入输出文件和失败日志
    with open(output_jsonl_path, 'a', encoding='utf-8') as out_file:
        # Step 4: 遍历所有疾病，逐个生成正则表达式
        for i, (disease_name, template_text) in tqdm(enumerate(disease_templates.items())):
            if disease_name in processed_diseases:
                print(f"✅ 跳过已处理疾病: {disease_name}")
                continue

            print(f"🔄 正在处理疾病: {disease_name}")

            try:
                regex_pattern = call_regex_generation(template_text, max_retries=5)
                result = {disease_name: regex_pattern}
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_file.flush()
                print(f"✅ 成功生成正则: {disease_name}")
            except Exception as e:
                print(f"❌ 生成失败: {disease_name}，错误: {str(e)}，跳过疾病: {disease_name}")
                failed.append(disease_name)
    print(failed)

def call_sample_generation(prompt: str, max_retries=3):
    """
    调用大模型生成正则表达式，带重试机制。
    """
    out = None
    error_msg = []
    frist_prompt = f'''
        帮我合成符合下面提供的正则表达式的数据，每个正则表达式需要合成4到8条，输出需要保证能被提供的正则表达式验证，同时输出格式限制为json格式的列表，如["样本1","样本2","样本3"]，回答不要包括其他内容，否则无法解析
        {prompt}
    '''
    
    for i in range(max_retries):
        full_prompt = frist_prompt
        
        if len(error_msg) > 0:
            print(f"[尝试 {i}/{max_retries}]次：{error_msg}")
            full_prompt += f"上一次生成不符合要求，请检查内容并重新生成，上一次生成:{out}，错误信息{error_msg}"
            error_msg = []
        
        response = call_dashscope_model(full_prompt)
        #out = f"{response}"

        try:
            parsed = json.loads(response)
            out = list(parsed)
        except (json.JSONDecodeError, TypeError):
            error_msg.append(f"格式错误：返回不是list格式，不能被json.loads解析")
        
        # if re.search(r'(?!(?<=\d)/(?=\d))/', response):
        #     error_msg.append(f"内容或格式错误：存在错误的/字符")
        
        if len(out) < 4:
            error_msg.append(f"内容错误：生成样本少于4个")
        
        seen = set()
        res=[]
        for idx, o in enumerate(out):
            if o in seen:
                error_msg.append(f"内容错误：生成样本存在重复")
                break
            else:
                seen.add(o)

            # 2. 判断是否符合正则
            if not re.search(prompt, o):
                error_msg.append(f"内容错误：第{idx}条内容'{o}'不满足正则，你需要仔细检测符号和内容是否符合正则模板,比如检测'-'是否正确，或者其他符号是否和模板匹配。请不要过多考虑语义问题，主要请让生成的内容能被解析")
                break
            
            res.append(o)  

        if len(error_msg) == 0:
            return res
    raise ValueError(f"{error_msg}")

def call_sample_generation2(prompt: str, max_retries=3):
    """
    调用大模型生成正则表达式，带重试机制。
    """
    out = None
    error_msg = []
    frist_prompt = r'''
我现在需要实现一个自动报告系统，应用场景如下：
医生可能会根据影像内容口述一些内容，比如，内中膜厚，内径5等零散的碎片化测量或观察特征，自动化报告系统需要根据这些碎片化内容检索对应病例模板并进行填充，最终生成格式化的规范模板。
任务背景：
我现在已经搭建好了这个系统，但是需要合成样本对系统生成和检索性能测试，因此，我需要你帮我完成合成样本的工作。
任务要求：
输入：
我会提供给你一个使用正则表达式描述的病例模板，用于生成参考以保证样本的准确性。
输出：
1.口语化内容:离散和口语化的关键词内容（不需要和模板完全匹配，口语表述的精炼和稀疏性是效率关键，也是检索难度所在，和模板高度重合的表述没有检索难度。）。
2.匹配模板的病例输出（这个输出则是和模板高度匹配，作为生成病例任务的gt）。
3.对于每个模板请生成4到8组样本用于系统测试，格式为[{"input":"口语化内容", "output": "完整病例内容"}，{"input":"口语化内容", "output": "完整病例内容"}，...]，回答不要包括多余部分，以便我使用json.loads()直接解析。

举一条例子：
输入模板："超声所见：左/右/双侧颈部\\((I|Ⅱ|Ⅲ|IV|V|VI)(?:[、，][I|Ⅱ|Ⅲ|IV|V|VI])*\\)区可见(一枚|多枚)淋巴结，呈(圆形|类圆形|分叶状)，大小\\d+(\\.\\d+)?cm×\\d+(\\.\\d+)?cm，厚径\\d+(\\.\\d+)?cm，L/T<2，包膜不光整，与周围组织界限不清晰，实质不规则，局限性增厚，内部回声不均匀，边缘可见(细小|点状)强回声，淋巴门呈缺失型。CDFI:淋巴结血管模式：(血管移位|血管迷行|局灶性无灌注|边缘血管)。频谱显示为动脉、静脉，Vmax=\\d+(\\.\\d+)?cm/s，RI=\\d+(\\.\\d+)?。超声提示：(左|右|双)侧颈部多发异常淋巴结肿大，考虑淋巴结转移癌可能。" 
输出内容：
1.口语化内容：右颈2区见1淋巴结，类圆形，大小1.2 0.8，厚0.5，包膜不光，界限不清，不规则，局限增厚，回声不均、强、点状，淋巴门缺失。可能淋巴结转移癌。
2.完整内容：超声所见：右颈部(II)区可见一枚淋巴结，呈类圆形，大小1.2cm×0.8cm，厚径0.5cm，L/T<2，包膜不光整，与周围组织界限不清晰，实质不规则，局限性增厚，内部回声不均匀，边缘可见点状强回声，淋巴门呈缺失型。CDFI:淋巴结血管模式：未提及 。频谱显示为 未提及 ，Vmax= 未提及 cm/s，RI= 未提及 。超声提示：右侧颈部多发异常淋巴结肿大，考虑淋巴结转移癌可能。" 
例子解释：
1.口语化内容：为了提高效率，省略了单位和不必要的连接词。
2.完整内容：符合正则表达式模板的要求，对于口语内容中未提及部分不做推测，使用 未提及 填充，保证生成准确性。

下面是输入模板：
'''+prompt
    
    for i in range(max_retries):
        full_prompt = frist_prompt
        
        if len(error_msg) > 0:
            print(f"[尝试 {i}/{max_retries}]次：{error_msg}")
            full_prompt += f"上一次生成不符合要求，请检查内容并重新生成，上一次生成:{out}，错误信息{error_msg}"
            error_msg = []
        
        response = call_dashscope_model(full_prompt)
        response = response.replace('```json', '')
        response = response.replace('```', '')
        response = response.strip()
        print(repr(response))
        #out = f"{response}"

        try:
            parsed = json.loads(response)
            out = list(parsed)
            if len(out) < 4:
                error_msg.append(f"内容错误：生成样本少于4个")
            res=out
        except json.JSONDecodeError:
            error_msg.append(f"格式错误：返回内容不能被json.loads解析,请检查格式以及字符是否存在错误")
        
        except Exception as e:
            error_msg.append(f"出现异常：{e}")
        
        # if re.search(r'(?!(?<=\d)/(?=\d))/', response):
        #     error_msg.append(f"内容或格式错误：存在错误的/字符")
        
        
        
        

            # # 2. 判断是否符合正则
            # if not re.search(prompt, o):
            #     error_msg.append(f"内容错误：第{idx}条内容'{o}'不满足正则，你需要仔细检测符号和内容是否符合正则模板,比如检测'-'是否正确，或者其他符号是否和模板匹配。请不要过多考虑语义问题，主要请让生成的内容能被解析")
            #     break  

        if len(error_msg) == 0:
            return res
    raise ValueError(f"{error_msg}")

from hypothesis import strategies as st
import re

def generate_samples_jsonl(
    disease_templates_path: str,
    output_jsonl_path: str,
):
    """
    从 JSONL 文件中读取疾病模板字典，批量生成样本
    """
    failed = []
    # Step 1: 从 JSONL 文件加载疾病模板
    disease_templates = {}
    with open(disease_templates_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            if isinstance(record, dict) and len(record) == 1:
                disease_name, template_text = next(iter(record.items()))
                disease_templates[disease_name] = template_text
            else:
                print(f"警告：无效的记录 {line.strip()}，跳过。")

    # Step 2: 加载已处理的疾病列表（用于断点续传）
    processed_diseases = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                processed_diseases.update(data.keys())

    # Step 3: 准备写入输出文件和失败日志
    with open(output_jsonl_path, 'a', encoding='utf-8') as out_file:
        # Step 4: 遍历所有疾病，逐个生成正则表达式
        for i, (disease_name, template_text) in tqdm(enumerate(disease_templates.items())):
            if disease_name in processed_diseases:
                print(f"✅ 跳过已处理疾病: {disease_name}")
                continue

            print(f"🔄 正在处理疾病: {disease_name}")

            try:
                samples = call_sample_generation2(template_text, max_retries=3)
                #compiled_ascii_regex = re.compile(template_text, flags=re.ASCII)
                #strategy = st.from_regex(compiled_ascii_regex, fullmatch=True)
                #samples = [strategy.example() for _ in range(10)]
                result = {disease_name: samples}
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_file.flush()
                print(f"✅ 成功生成正则: {disease_name}")
            except Exception as e:
                print(f"❌ 生成失败: {disease_name}，错误: {str(e)}，跳过疾病: {disease_name}")
                failed.append(disease_name)
    print(failed)

def clean_template_jsonl(
    disease_templates_path: str,
    output_jsonl_path: str,
):
    """
    从 JSONL 文件中读取疾病模板字典，批量生成样本
    """
    failed = []
    # Step 1: 从 JSONL 文件加载疾病模板
    disease_templates = {}
    with open(disease_templates_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            if isinstance(record, dict) and len(record) == 1:
                disease_name, template_text = next(iter(record.items()))
                print(template_text)
                template_text=template_text.replace('\\s*','')
                template_text=template_text.replace(',','，')
                print(template_text)
                disease_templates[disease_name] = template_text
            else:
                print(f"警告：无效的记录 {line.strip()}，跳过。")

    # Step 3: 准备写入输出文件和失败日志
    with open(output_jsonl_path, 'a', encoding='utf-8') as out_file:
        # Step 4: 遍历所有疾病，逐个生成正则表达式
        for i, (disease_name, template_text) in tqdm(enumerate(disease_templates.items())):
            try:
                result = {disease_name: template_text}
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_file.flush()
            except Exception as e:
                print(f"❌ 生成失败: {disease_name}，错误: {str(e)}，跳过疾病: {disease_name}")
                failed.append(disease_name)
    print(failed)
    
    
    

#process_and_save_to_jsonl(input_path, os.path.join(current_dir, 'temp.jsonl'))
#generate_regex_jsonl(os.path.join(current_dir, 'temp.jsonl'), output_path)
generate_samples_jsonl('/root/autodl-tmp/RAG/SpeechAutoReporter/templates/超声报告模板/超声报告模板.jsonl', os.path.join(current_dir, 'sample2.jsonl'))
#clean_template_jsonl(output_path, os.path.join(current_dir, 'new.jsonl'))

