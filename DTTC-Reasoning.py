from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from datasets import load_dataset
import re
import torch
from sympy import parse_expr 
import time

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "openai/gsm8k"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CANDIDATES = 6
FORMAT_WEIGHT = 0.3
CORRECT_WEIGHT = 0.7
PARAM_POOL = [
    {'temp': 0.6, 'top_p': 0.9, 'rep_penalty': 1.2},

    {'temp': 0.5, 'top_p': 0.875, 'rep_penalty': 1.3},
    
    {'temp': 0.7, 'top_p': 0.925, 'rep_penalty': 1.1},
    
    {'temp': 0.8, 'top_p': 0.95, 'rep_penalty': 1.05},
    
    {'temp': 0.4, 'top_p': 0.85, 'rep_penalty': 1.4},
    {'temp': 0.9, 'top_p': 0.98, 'rep_penalty': 1.0}
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token

# 歧义句型映射表
AMBIGUOUS_PATTERNS = {                  
    r"\b(\d+)\s+sprints?\s+(\d+)\s+times\s+a\s+week\b": 
        "每周进行{m1}次训练，每次训练跑{m0}次冲刺跑",
    r"\b(\d+)\s+packs?\s+of\s+(\d+)\b": 
        "以每包{m1}个的方式购买{m0}包",
    r"\b(\d+)\s+boxes?\s+a\s+day\b": 
        "每天使用{m0}盒，每盒装有",
    r"\b(\d+)\s+pages?\s+per\s+minute\b": 
        "每分钟持续打印{m0}页",
    r"\b(\d+)\s+shelves?,\s+each\s+holding\s+(\d+)\b": 
        "有{m0}个书架，每个书架上有{m1}本书"
}
AMBIGUOUS_PATTERNS.update({
    # 处理财务类问题的时间维度歧义
    r"\b(\d+)\s+to\s+(?:plant|buy)\b": 
        "需一次性支付{m0}（单位：美元）初始成本",
    r"\b(\d+)\s+a\s+year\s+to\s+(?:water|maintain)\b": 
        "每年持续支出{m0}（单位：美元）维护费用",
    r"\bstarts?\s+earning\s+money\b": 
        "累计总收入超过初始投入与维护费用总和所需年数",
    r"\bgrow\s+(\d+)\s+lemons\b": 
        "每年稳定产出{m0}个柠檬（视为持续产能）",
    r"\bsell\s+for\s+\$([\d.]+)\s+each\b": 
        "每个产品售价为{m0}美元（价格恒定）"
})
AMBIGUOUS_PATTERNS.update({
    # 出生时间绝对差（原题中"born 6 years before"）
    r"\bborn\s+(\d+)\s+years?\s+(before|earlier than)\b": 
        "出生年份绝对相差{m0}年",
    
    # 生育年龄参照系（原题中"had a son at the age of 23"）
    r"\bhad\s+a\s+child\s+at\s+the\s+age\s+of\s+(\d+)\b": 
        "生育事件发生时父母年龄为{m0}岁（需用当前年龄-{m0}=子女生育年数）",
    
    # 动态年龄差陷阱（易错点：年龄差恒定）
    r"\b(\d+)\s+years?\s+(older|younger)\s+than\b": 
        "两人年龄差恒为{m0}岁",
    
    # 隐性时间参照系（原题中"Samantha is now 31"）
    r"\bIf\s+(\w+)\s+is\s+now\s+(\d+)\b": 
        "当前时间点作为所有年龄计算的基准"
})
AMBIGUOUS_PATTERNS.update({
    # 时间比较句型（错误案例：Lee和Gerald赛跑问题）
    r"\b(\d+)\s+seconds?\s+faster/slower\s+than\b": 
        "时间差为{m0}秒（非速度比例，直接加减时间值）",
        
    # 倍数分配句型（错误案例：Doubtfire家族小猫问题）
    r"\b(thrice|triple|three times)\s+the\s+number\s+of\b": 
        "数量是原先的三倍（直接乘3计算）",

    # 新生与收养数量关系（错误案例：Doubtfire家族小猫问题）
    r"\bhad\s\b(thrice|triple|three times)+\s+adopted\s+kittens\b": 
        "新生小猫数量=已收养数量×3（不包含原始收养量）",

    # 复合动物来源统计
    r"\badopted\s+kittens\s+plus\s+newborn\s+kittens\b": 
        "总数量=收养数量+新生数量（需确认是否包含原始收养量）",

    # 复合增长率句型（错误案例：大理石年增长问题）
    r"\bincreases?\s+by\s+(\d+)%?\s+of\s+the\s+original\s+price\b": 
        "按原价的{m0}%金额增长（线性叠加，非复利计算）",
    
    # 路径往返句型（错误案例：Julia划船漏水问题）
    r"\bback\s+and\s+forth\s+(\d+)\s+times?\b": 
        "往返{m0}次（总距离=单程距离×2×次数）",
        
    # 分配平均数句型（错误案例：Russell宠物店仓鼠问题）
    r"\bin\s+equal\s+groups?\s+across\s+(\d+)\s+cages?\b": 
        "平均分配到{m0}个容器中（总数量÷容器数）",
    
    # 隐含单位转换句型（错误案例：博物馆昆虫腿数问题）
    r"\b(\d+)\s+yard\s+line\s+and\s+back\b": 
        "前往{m0}码线并返回（总距离为{m0}×2码）",
    
    # 非连续操作句型（错误案例：Griffin薯条问题）
    r"\btwice\s+as\s+many\s+as\s+(\w+)\s+had\s+taken\b": 
        "本次取走量是{m0}上次取走量的两倍（独立事件不累计）",

    # 单位热量密度转换（薯片卡路里问题）
    r"\b(\d+)g\s+bag\s+has\s+(\d+)\s+servings\b": 
        "总重量{m0}g对应{m1}份（每份重量={m0}÷{m1}g）",

})

AMBIGUOUS_PATTERNS.update({
    # 倍数加减句型（错误案例：Walmart驱逐顾客问题）
    r"\bfour times that many minus (\d+)\b":
    "数量等于原数量的四倍减去{m0}（非单纯倍数关系）",

    # 价格节省句型（错误案例：流媒体套餐问题）
    r"\bsaves (\d+)% for bundling\b": 
        "捆绑购买时享受{m0}%折扣（按总价计算折扣）",

    # 时间节省换算句型（错误案例：空调节能问题）
    r"\breduces the time by (\d+) hours\b": 
        "减少{m0}小时使用时间（按每小时能耗计算总节省）",

    # 隐含步骤顺序句型（错误案例：啦啦队金字塔高度问题）
    r"\bstand on top of the (\d+)\b": 
        "每层人员独立站立（高度直接叠加，非平均高度）",

    # 复合单位转换句型（错误案例：卧室周长问题）
    r"\b(\d+) yards?\b": 
        "{m0}码（需转换为英尺，1码=3英尺）",

    # 特殊量词句型（错误案例：酒店洗衣总量问题）
    r"\btwice as many (\w+) as (\w+)\b": 
        "{m0}数量是{m1}的两倍（直接乘2计算）",

    # 动态数量变化句型（错误案例：消防员鞋子问题）
    r"\bgets rid of (\d+) shoes\b": 
        "淘汰{m0}只鞋（注意单位，1双=2只）",

    # 复合收益计算句型（错误案例：柠檬摊利润问题）
    r"\b(\d+) cups per hour\b": 
        "每小时售出{m0}杯（需结合总产量计算营业时长）",

    # 隐性包含关系句型（错误案例：超级票套餐问题）
    r"\bincludes rights to watch any movie\b": 
        "套餐费用已包含所有权益（无需额外计算组件价格）",

    # 位置变化句型（错误案例：赛跑名次问题）
    r"\bfell back (\d+) spots\b": 
        "名次后退{m0}位（直接数字加减计算位置）"
})

AMBIGUOUS_PATTERNS.update({
    # 木材产量倍数关系（Frederick冰棍棒问题）
    r"\b(\d+)\s+sticks\s+from\s+a\s+([\dx]+)\s+piece\b": 
        "每{m1}规格木材可生产{m0}根棒（产量与规格正相关）",
    
    # 通勤次数句型（Janet公交问题）
    r"\btwo\s+bus\s+trips\s+five\s+days\s+a\s+week\b": 
        "每周5天×每天2次=10次乘车（次数与天数需相乘）",
    
    # 热量缺口计算句型（Andy减肥问题）
    r"\bburn\s+(\d+)\s+calories?\s+to\s+lose\s+a\s+pound\b": 
        "每减1磅需{m0}卡路里缺口（总缺口=目标磅数×{m0}）",
    
    # 包装商品数量（Tara画布包问题）
    r"\b(\d+)\s+packs?\s+of\s+(\d+)\s+canvas\s+bags\b": 
        "以每包{m1}个的方式购买{m0}包（总数量={m0}×{m1}）",
    
    # 动态库存变化（面包店问题）
    r"\bsold\s+(\d+)\s+loaves\s+in\s+the\s+morning\s+and\s+(\d+)\s+in\s+the\s+afternoon\b": 
        "先减早间销量{m0}，再减午后销量{m1}，最后处理退货",
    
    # 替代消费计算（Tom耳机问题）
    r"\bhow\s+many\s+more\s+CDs\s+if\s+not\s+buy\s+the\s+headphone\b": 
        "总预算÷单品价格-已购数量=可多购数量",
    
    # 例外时间统计（Josh健身问题）
    r"\b(\d+)\s+hours\s+every\s+week\s+except\s+on\s+some\s+occasions\b": 
        "基准时长×总周数 - 例外周数×基准时长 + 例外周额外时长",
    
    # 乘客增减倍数（公交问题）
    r"\b(\d+)\s+times\s+as\s+many\s+people\s+as\s+the\s+number\s+who\s+got\s+off\b": 
        "上车人数=下车人数×{m0}（直接乘法计算）",
    
    # 跨时间点年龄差（Jame年龄问题）
    r"\bIn\s+(\d+)\s+years\s+his\s+cousin\s+will\s+be\s+(\d+)\s+years\s+younger\s+than\s+twice\s+his\s+age\b": 
        "建立方程：C+{m0}=2×(当前年龄+{m0}) - {m1}",
    
    # 小费与分摊计算（晚餐账单问题）
    r"\bevenly\s+split\s+the\s+check\s+but\s+pay\s+an\s+additional\s+(\d+)%\s+tip\b": 
        "总费用=原价×(1+{m0}%)，再平分",
    
    # 复合倍数摄入（咖啡问题）
    r"\b(\d+)\s+times\s+the\s+amount\s+she\s+drinks\b": 
        "摄入量=基准量×{m0}（无需累加基准量）",
    
    # 年龄对应商品（James蜡烛问题）
    r"\bOne\s+is\s+(\d+)\s+and\s+the\s+other\s+is\s+(\d+)\s+years\s+younger\b": 
        "需为每个年龄单独购买对应数量商品（非合并计算）",
    
    # 复合成本利润（食品卡车问题）
    r"\bsource\s+their\s+bread\s+for\s+\$([\d.]+)\s+a\s+loaf\s+and\s+each\s+loaf\s+makes\s+(\d+)\b": 
        "单位成本=面包成本/{m1} + 奶酪成本/{m1}",
    
    # 跨年动态成本（John癌症研究）
    r"\bevery\s+month\s+after\s+those\s+first\s+(\d+)\s+took\s+(\d+)%?\s+more\s+funding\b": 
        "后续月份成本=初始月成本×(1+{m1}%)^月份数",
    
    # 套装组件价格（Bill购车问题）
    r"\bleather\s+seats\s+are\s+one-third\s+the\s+cost\s+of\s+the\s+king\s+cab\b": 
        "组件价格=基准组件价格×分数（直接乘除计算）",
    
    # 跨平台点赞统计（Fishio点赞问题）
    r"\b(\d+)\s+times\s+as\s+many\s+as\s+the\s+initial\s+number\s+of\s+likes\b": 
        "新点赞数=初始点赞数×{m0}（非叠加计算）"
})

AMBIGUOUS_PATTERNS.update({
# 动态数量翻倍句型（错误案例：James狗玩具问题）
r"\btwice as many more dogs than when he left\b":
"数量增加为离开时的两倍（一次性变化，非持续翻倍）",

# 复合商品总价句型（错误案例：Manolo糖果问题）
r"\b(\d+) (lollipops|candies) and (\d+) (lollipops|candies) that cost \$([\d.]+)\b": 
    "购买{m0}个{m1}和{m2}个{m3}，总价为{m4}美元（需分别计算单价）",

# 代际倍数关系句型（错误案例：Great Grandma Jones家庭问题）
r"\beach of (?:her|his) (children|grandchildren) has (\d+) (?:children|babies) of their own\b": 
    "每个{m1}有{m2}个孩子（代际数量按乘法计算）",

# 净节省计算句型（错误案例：Ron修路问题）
r"\bsaves money by fixing the pothole\b": 
    "节省金额=避免的损失 -（材料成本 + 罚款）",

# 分数剩余量计算句型（错误案例：Bryce披萨问题）
r"\bate ([\d/]+) of their pizzas\b": 
    "吃掉披萨的{m0}（剩余片数=总片数×(1 - {m0}）",

# 单位节省扩展句型（错误案例：番茄商贩问题）
r"\bsaves \$\d+\.\d+ a week\b": 
    "每周节省=每天节省×7（需先计算每天单位节省×数量）",

# 分组数量变化句型（错误案例：学校分组问题）
r"\bseparated into (\d+) groups of equal size\b.*requires (\d+) groups\b": 
    "从{m0}组变为需要{m1}组（保持每组人数不变，计算需新增组数=总人数/原组人数 - 现有组数）"
})
AMBIGUOUS_PATTERNS.update({
    # 包含性人数计算（错误案例：Jenna室友电费问题）
    r"\b(\d+) roommates?\b": 
        "共同居住人数为{m0}人（包含本人在内的总人数需+1计算）",
    
    # 复合容器数量统计（错误案例：Jeff餐具问题）
    r"\bhow many glasses and plates\b": 
        "需分别统计玻璃器皿和餐具数量（非求和计算）",
    
    # 百分比兑换操作（错误案例：Lorraine贴纸交易）
    r"\btrades (\d+)% of (?:her|his) (\w+) for (\w+)\b": 
        "将{m1}的{m0}%按兑换规则转换为{m2}（需保持单位一致性）",
    
    # 多阶段倍数方程（错误案例：屋顶瓦片分配问题）
    r"\bthird needing double the first\b": 
        "第三个变量与第一个变量呈2倍关系（需建立x + 2x + 4x型方程）",
    
    # 分段次数计算（错误案例：骑行次数问题）
    r"\bon two other days, they ride twice the times they do on usual days\b": 
        "额外天数中的次数=常规次数×2（总次数=常规天数×常规次数 + 额外天数×2倍次数）",
    
    # **复合剩余分配（错误案例：棉花糖问题）**
    r"\bhow many S'mores can each kid have with the marshmallows left\b": 
        "剩余总量需平均分配（每人数量=剩余总数÷人数）",
    
    # **隐含时间跨度（错误案例：酒店住宿时长问题）**
    r"\bfrom (\d+):\d+ (?:AM|PM) until (\d+):\d+ (?:AM|PM)\b": 
        "时间跨度为{m1}-{m0}小时（需计算24小时制时长）",
    
    # **动态价格结构（错误案例：水管工收费问题）**
    r"\b\$(\d+) per hour, or part thereof\b": 
        "不足1小时按1小时计费（总时长向上取整计算）",
    
    # **集合操作顺序（错误案例：海盗挖洞问题）**
    r"\bfour times as many holes by then as it did at the end of the first day\b": 
        "第四天总洞数=第一天结束时洞数×4（需包含填洞操作后的净值）",
    
    # **复合折扣计算（错误案例：图书退货问题）**
    r"\brefund (\d+)% of the (?:item’s|book’s) purchase price\b": 
        "退款金额=原价×{m0}%（需扣除运费等附加成本）"
})
AMBIGUOUS_PATTERNS.update({
    # 时间阶段连续累加（错误案例：Jordan蛋糕问题）
    r"\b(\d+)\s+hours?\s+to\s+cool\s+and\s+an\s+additional\s+(\d+)\s+minutes?\b": 
        "冷却{m0}小时与装饰{m1}分钟需连续累加（总时长=各阶段时间简单求和）",
    
    # 百分比线性衰减（错误案例：移动电源耗电问题）
    r"\blosing\s+(\d+)%\s+of\s+the\s+total\s+capacity\s+each\s+hour\b": 
        "每小时减少总容量的{m0}%（按线性递减计算，非复利公式）",
    
    # 集合数量均分（错误案例：毕业门票分配问题）
    r"\btickets\s+split\s+equally\s+among\s+(\d+)\s+graduates?\b": 
        "剩余票数直接除以毕业生数{m0}（无需考虑余数或进位）",
    
    # 复合入场规则（错误案例：酒店客人计算问题）
    r"\btwice\s+as\s+many\s+people\s+checked\s+in\s+as\s+those\s+who\s+opted\s+for\b": 
        "新入场人数=参照组人数×2（直接倍数计算）",
    
    # 动态资源消耗（错误案例：移动电源充电问题）
    r"\badded\s+(\d+)%\s+of\s+them\s+to\s+her\s+remaining\s+clutch\b": 
        "添加比例为剩余总量的{m0}%（按当前剩余量计算）",
    
    # 复合年龄差计算（错误案例：Liam年龄问题）
    r"\btwo\s+years?\s+ago,\s+.*age\s+was\s+twice\s+the\s+age\s+of\b": 
        "建立方程：当前年龄-2 = 2×(对方当前年龄-2)",
    
    # 分组容器容量（错误案例：书架容量问题）
    r"\beach\s+of\s+the\s+middle\s+(\d+)\s+shelves?\s+can\s+hold\s+(\d+)\b": 
        "中层容量={m0}层×{m1}本/层，顶层与底层单独计算",
    
    # 复合增长率误解（错误案例：鳄梨树产量问题）
    r"\bproduces\s+(\d+)\s+times\s+the\s+initial\s+amount\b": 
        "产量=初始值×{m0}（非累加计算）",
    
    # 动态数量替换（错误案例：牛顿苹果问题）
    r"\bcausing\s+(\d+)\s+more\s+to\s+fall\s+out\s+of\s+the\s+tree\b": 
        "苹果总量变化=当前数量+{m0}（需考虑前序操作后的当前值）"
})
def preprocess_question(question):
    def replacer(match):
        params = match.groups()
        pattern = AMBIGUOUS_PATTERNS[match.re.pattern]
        return pattern.format(**{f'm{i}': p for i, p in enumerate(params)})
    
    # 应用模式替换
    processed = question
    for pattern in AMBIGUOUS_PATTERNS:
        processed = re.sub(pattern, replacer, processed, flags=re.IGNORECASE)
    
    return f"""请用中文逐步推理并给出答案（用\\boxed{{}}标记）：
            问题：{processed}
            **要求**：
            1. 使用中文完整表述解题过程；
            2. 最终答案必须用\\boxed{{}}包裹；
            3. 避免重复问题描述；
            4. 避免重复写出答案"""
           

# 加载数据集
def load_gsm8k():
    data_path = "/root/.cache/huggingface/datasets/openai___gsm8k/openai___gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee/gsm8k-test.arrow"
    dataset = load_dataset(
        "arrow",
        data_files={"test": data_path},
        split="test"
    )
    questions = [preprocess_question(ex["question"]) for ex in dataset]
    answers = [ex["answer"] for ex in dataset]
    return questions, answers

questions, ground_truths = load_gsm8k()

# 答案解析函数（增强鲁棒性）
def parse_answer(text):
    # 优先提取最后一个\boxed{}内容（增强中文支持）
    latex_matches = re.findall(r'\\boxed{([^}]+)}', text)
    if latex_matches:
        try:
            # 处理中文数字和逗号
            raw = latex_matches[-1].replace('\\text{dollars}', '')\
                                   .replace('，', '')\
                                   .replace(',', '')
            # 直接转换为整数（避免浮点误差）
            return int(float(parse_expr(raw)))
        except:
            pass
    # 优先尝试解析LaTeX表达式
    latex_match = re.search(r'\\boxed{([^}]+)}', text)
    if latex_match:
        try:
            expr = parse_expr(latex_match.group(1).evalf())
            return float(expr)
        except:
            pass  # 失败时回退到原有逻辑

    chain_of_thought = re.findall(r'\d+\.?\d*', text)
    if chain_of_thought:
        try:
            return float(chain_of_thought[-1])  # 假设最后一步为答案
        except:
            pass 

    patterns = [
        r'\\boxed{([^}]+)}',  # 匹配LaTeX的\boxed{}中的任何内容
        r'Final Answer:\s*([-+]?\d+\.?\d*|[-+]?\d+/\d+|[-+]?\d+\s+\d+/\d+)',  # 处理分数
        r'Answer:\s*([-+]?\d+\.?\d*|[-+]?\d+/\d+)',
        r'答案\s*[:：]\s*([-+]?\d+\.?\d*|[-+]?\d+/\d+)',
        r'x\s*=\s*([-+]?\d+\.?\d*|[-+]?\d+/\d+)',
        r'\$([-+]?\d+\.?\d*)',
        r'\$?(-?\d{1,3}(?:,\d{3})*\.?\d*)(?:\s*(?:dollars|cups|GB|miles|mph|years))?',  # 带单位的数值
        r'(?<=\$)\d+\.?\d*',  # 美元数值
        r'\b\d+/\d+\b',      # 分数
        r'\b\d+\s+\d+/\d+\b' # 混合分数
    ]
    patterns += [
        r'(\d+)\s*年',  # 匹配“12年”
        r'after\s*(\d+)\s*years',  # 匹配英文表述
        r'[-+]?\d+\s*[/÷]\s*\d+'  # 加强分数解析（如 90/7.5）
    ]
    for pattern in patterns:
        # 处理单位（如"20 cups" → 20）
        unit_pattern = r'(\d+\.?\d*)\s*(cups|dollars|GB)'
        match = re.search(unit_pattern, text) 
        if match:
            value = match.group(1)
            try:
                # 处理分数
                if '/' in value:
                    numerator, denominator = value.split('/')
                    return float(numerator) / float(denominator)
                # 处理可能包含逗号的数字（如1,000）
                value = value.replace(',', '')
                # 处理混合分数（如 2 3/4 → 2.75）
                if ' ' in value and '/' in value:
                    whole, fraction = value.split(' ')
                    numerator, denominator = fraction.split('/')
                    return float(whole) + float(numerator)/float(denominator)
                # 处理科学计数法（如 1e3）
                if 'e' in value.lower():
                    return float(value)
            except:
                continue
    # 最后尝试直接搜索数字
    match = re.search(r'[-+]?\d+\.?\d*', text)
    return float(match.group()) if match else None


# 思路切换触发词
SWITCH_TOKENS = [
    '另一种方法', 'alternatively', '或者', '换一种思路',
    '但是', '另一方面', '然而',
]

# TIP Logits处理器
class TIPLogitsProcessor(LogitsProcessor):
    def __init__(self, switch_token_ids, alpha=3.0, beta=150):
        self.switch_token_ids = switch_token_ids
        self.alpha = alpha
        self.beta = beta
        self.current_thought_start = 0

    def __call__(self, input_ids, scores):
        last_token = input_ids[0][-1].item()
        if last_token in self.switch_token_ids:
            self.current_thought_start = input_ids.shape[-1]
        
        current_position = input_ids.shape[-1]
        alpha = self.alpha * (1 - (current_position / (current_position + self.beta)))
        if current_position < self.current_thought_start + self.beta:
            for token_id in self.switch_token_ids:
                scores[:, token_id] -= alpha
        return scores

def validate_answer(pred_num, gt_num):
    """严格整数匹配"""
    if isinstance(gt_num, int):
        return int(pred_num) == gt_num
    """增强数值验证逻辑"""
    if pred_num is None or gt_num is None:
        return False
    
    # 处理不同数量级的情况
    relative_err = abs(pred_num - gt_num) / (abs(gt_num) + 1e-5)
    absolute_err = abs(pred_num - gt_num)
    
    # 根据问题类型动态调整阈值
    if gt_num > 100:  # 大数值允许1%误差
        return relative_err < 0.01
    elif gt_num < 1:  # 小数允许更高精度
        return absolute_err < 0.02
    else:  # 常规数值严格匹配
        return absolute_err < 0.1 or relative_err < 0.005


# 增强格式验证函数
def format_scoring(text):
    score = 0
    # 检查是否包含答案框
    if re.search(r'\\boxed{', text):
        score += 0.4
    # 检查是否使用中文
    if len(re.findall(r'[\u4e00-\u9fff]', text)) > 10:
        score += 0.3
    # 检查重复问题描述
    if "问题：" not in text.split("**要求**")[0]:
        score += 0.2
    # 检查答案重复
    if text.count("答案") + text.count("Answer") > 1:
        score -= 0.2
    return max(0, min(1, score))

# 增强奖励计算
def calculate_reward(pred_text, gt_number):
    pred_number = parse_answer(pred_text)
    # 正确性评分
    correct_score = 1 if validate_answer(pred_number, gt_number) else 0
    # 格式评分
    format_score = format_scoring(pred_text)
    return CORRECT_WEIGHT * correct_score + FORMAT_WEIGHT * format_score



# 生成答案
def batch_generate(batch_questions, param):
    switch_token_ids = tokenizer.convert_tokens_to_ids(SWITCH_TOKENS)
    inputs = tokenizer(
        batch_questions,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=param['temp'],
            top_p=param['top_p'],
            repetition_penalty=param['rep_penalty'],
            do_sample=True,
            num_return_sequences=1,
            num_beams=4,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[TIPLogitsProcessor(switch_token_ids)],
            use_cache=True
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]



# 评估准确率
correct = 0
error = 0
results = []
start_time = time.time()

# 初始化全局队列系统
from collections import deque
question_pool = deque([{
    'idx': idx,
    'question': q,
    'gt': parse_answer(gt),
    'attempted_params': [],  # 已尝试的参数索引列表
    'candidates': []         # 生成的候选答案
} for idx, (q, gt) in enumerate(zip(questions, ground_truths)) if parse_answer(gt) is not None])

# 动态队列处理器
def process_queue():
    global correct, error
    
    current_batch = []
    # 填充当前批次（优先处理未完成的问题）
    while question_pool and len(current_batch) < BATCH_SIZE:
        item = question_pool.popleft()
        
        # 确定当前要使用的参数索引
        next_param_idx = len(item['attempted_params'])
        if next_param_idx < len(PARAM_POOL):
            current_batch.append(item)
        else:
            # 参数用尽仍未解决，记录错误
            error += 1
            results.append(("✗", item['candidates'][-1], item['gt']))
    
    if not current_batch:
        return False
    
    # 批量生成（每个问题使用下一个参数）
    param_indices = [len(item['attempted_params']) for item in current_batch]
    params_to_use = [PARAM_POOL[idx] for idx in param_indices]
    
    # 按参数分组处理（优化生成效率）
    param_groups = {}
    for i, param in enumerate(params_to_use):
        param_key = tuple(param.values())
        if param_key not in param_groups:
            param_groups[param_key] = []
        param_groups[param_key].append((i, current_batch[i]))
    
    # 并行生成各组
    processed_items = []
    for param, group in param_groups.items():
        batch_questions = [item['question'] for _, item in group]
        batch_answers = batch_generate(batch_questions, dict(zip(PARAM_POOL[0].keys(), param)))
        
        for (orig_idx, item), ans in zip(group, batch_answers):
            item['attempted_params'].append(param_indices[orig_idx])
            item['candidates'].append(ans)
            
            # 立即验证
            pred_num = parse_answer(ans)
            if validate_answer(pred_num, item['gt']):
                correct += 1
                results.append(("✓", ans, item['gt']))
            else:
                processed_items.append(item)
    
    # 未解决的问题重新入队（保持处理顺序）
    question_pool.extendleft(reversed(processed_items))
    
    return True

# 主循环
while process_queue():
    print(f"进度: {correct+error}/{len(questions)} 准确率: {correct/(correct+error):.2%}")

# 处理剩余问题（参数用尽的情况）
while question_pool:
    item = question_pool.popleft()
    error += 1
    results.append(("✗", item['candidates'][-1], item['gt']))

# 最终统计（保持原有输出格式）
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"\n基准测试结果（GSM-8K）:")
print(f"正确处理: {correct} 错误处理: {error}")
print(f"准确率: {correct/(correct+error):.2%}")
print(f"总耗时: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
# 打印错误分析示例
error_samples = [r for r in results if r[0] == "✗"]
# print("\n错误分析:")
# for mark, pred, gt in error_samples[:1]:
#     print(f"预测: {pred}...")
#     print(f"参考答案: {gt}...\n")


with open('error.txt', 'w', encoding='utf-8') as f:
    for mark, pred, gt in error_samples:
        f.write((f"预测: {pred}..."))
        f.write(f"参考答案: {gt}...\n")
    
truth_samples = [r for r in results if r[0] == "✓"]

with open('truth.txt', 'w', encoding='utf-8') as f:
    for mark, pred, gt in truth_samples:
        f.write((f"预测: {pred}..."))
        f.write(f"参考答案: {gt}...\n")
    