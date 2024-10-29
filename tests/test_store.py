from qwergpt.store import DocumentStore


# 创建 DocumentStore 实例
doc_store = DocumentStore()

# 插入单个文档
doc_store.insert("中国是一个拥有悠久历史和丰富文化的国家。")

# 批量插入多个文档
docs = [
    "北京是中国的首都，也是政治、文化、国际交往中心。",
    "上海是中国最大的经济中心城市，也是国际金融中心。",
    "广州是中国南方重要的经济、文化和交通中心。",
    "深圳是中国改革开放的窗口，高新技术产业发达。"
]
doc_store.batch_insert(docs)

# 查询相似文档
query = "中国的大城市"
results = doc_store.query(query, k=3)

print("查询:", query)
print("最相似的3个文档:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc}")
