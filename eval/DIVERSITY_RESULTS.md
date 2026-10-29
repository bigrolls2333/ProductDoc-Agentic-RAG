# 扩充开发评测集：商品与问题多样性

当前版本共有130道已通过AI来源审核的开发题，其中保留原有25题，新增105题；另有21道新增题待审。

**这是来源约束的合成开发集，不是人工金标或独立盲测集；题量增加不等于获得同等数量的独立真实用户问题。**

## 覆盖变化

| 维度 | 原有题 | 新增已审核题 | 合计 |
| --- | ---: | ---: | ---: |
| 题数 | 25 | 105 | 130 |
| 涉及商品数 | 7 | 27 | 28 |
| 多商品题 | 4 | 13 | 17 |
| 多轮指代题 | 0 | 6 | 6 |
| 含表格种子证据 | 未补标签 | 31 | 至少 31 |
| 可回答题 | 20 | 105 | 125 |
| 无答案题 | 5 | 0 | 5 |
| 人工核验题 | 0 | 0 | 0 |

## 任务类型

| 类型 | 原有题 | 新增题 | 合计 |
| --- | ---: | ---: | ---: |
| constraint_check | 0 | 15 | 15 |
| multi_aspect | 0 | 8 | 8 |
| multi_item | 4 | 13 | 17 |
| multi_turn | 0 | 6 | 6 |
| no_answer | 5 | 0 | 5 |
| operation_step | 4 | 24 | 28 |
| product_fact | 4 | 6 | 10 |
| table_lookup | 0 | 17 | 17 |
| troubleshooting | 8 | 16 | 24 |

## 商品分布

| 商品标准名 | 原有题 | 新增题 | 合计 |
| --- | ---: | ---: | ---: |
| AOLYNKCB304N网桥 | 1 | 3 | 4 |
| H3CER2100企业级路由器 | 3 | 6 | 9 |
| H3CLA2608室内无线网关 | 11 | 4 | 15 |
| H3CMER系列路由器 | 0 | 6 | 6 |
| H3CMSR-XS系列路由器 | 0 | 5 | 5 |
| H3CMSRWINET系列云简路由器 | 0 | 4 | 4 |
| H3CMSR系列开放多业务路由器 | 0 | 3 | 3 |
| H3CNR-1200W企业级无线宽带路由器 | 3 | 6 | 9 |
| HUAWEIB3-430笔记本电脑 | 0 | 5 | 5 |
| HUAWEIB5-420笔记本 | 0 | 6 | 6 |
| HUAWEIB7-410笔记本电脑 | 0 | 3 | 3 |
| HUAWEIMATESTATIONB515台式电脑 | 0 | 5 | 5 |
| HUAWEIPIXLABB5打印机 | 5 | 5 | 10 |
| LENOVOLJ2268激光打印机 | 0 | 3 | 3 |
| LENOVOM7208WPRO激光多功能一体机 | 1 | 4 | 5 |
| LENOVOM7268激光多功能一体机 | 0 | 3 | 3 |
| PANTUMP3030黑白激光打印机 | 5 | 0 | 5 |
| RS-PRORS-12数字万用表 | 0 | 3 | 3 |
| Z26打印机 | 0 | 6 | 6 |
| 华为B3-241H显示器 | 0 | 3 | 3 |
| 华为C5E平板 | 0 | 3 | 3 |
| 华为C7平板 | 0 | 6 | 6 |
| 华为平板C5平板 | 0 | 6 | 6 |
| 华为擎云G540笔记本电脑 | 0 | 5 | 5 |
| 奔图P3500黑白单功能激光打印机 | 0 | 3 | 3 |
| 打印机 | 0 | 5 | 5 |
| 绿米小米多功能网关网关 | 0 | 3 | 3 |
| 联想Z26MIC打印机 | 0 | 4 | 4 |

## 数据如何产生

- 保持原有语料与25道已审核题不变，从全部28份手册的分散片段出题；按参数/表格、操作、约束或多方面、故障或指代分配任务，再增加同类商品对比。
- 查询生成看文档，不看检索排名、系统答案或策略分数。新题包含正式、口语和上下文表达，并非只给旧问题换同义词。
- 种子引文必须逐字存在于chunk，并能在原始PDF文本中按明确规范化规则定位；不匹配的出题任务只做来源修复，不按检索效果挑题。
- 另一轮模型调用对目标商品全部chunks逐条标注相关性，每个正相关判断都要给原文引文；再次核验PDF。原文无法核验的正相关保留未知，不能自动改成负例。
- 独立参考审核从这些原文事实提取最小参考要点；要求可完整回答、多商品证据齐全、目标文档标注覆盖至少90%，否则保持待审。
- 其余商品的文档只在明确的商品手册任务范围下记为不相关；因此分数解释依赖商品范围。
- 新题全部放在dev，并保守共享分组，不拆成随机训练/测试集；多轮题另存审核后的standalone retrieval_query。
- 旧题没有重新补表格种子标签，不能把其未记录数当成不含表格；本轮未增加无答案题，仍保留原有5题。
- 版本还保留原有246道未审核题；它们与新增待审题均不进入有效评测，所以工作台总题数会高于本报告的有效题数。
- 合成问题精确重复数：0；可回答题中完全相同的参考引文集合复用数：0。共享手册与相似事实仍存在相关性，不能当作独立统计样本。

## 新题示例

- VAL-DIV-003（constraint_check）：H3CMSR-XS系列路由器配置一对一映射时，如果设备上只有一个公网IP地址，文档有什么建议？另外配置完成后还需要在页面上做什么操作才能生效？
- VAL-DIV-007（multi_aspect）：联想打印机的墨粉盒和感光鼓按ISO/IEC 19752测试页算，平均寿命各是多少页？
- VAL-DIV-114（multi_item）：这两款笔记本在开机时进入F10恢复出厂的操作前提有什么相同要求？
- VAL-DIV-020（multi_turn）：历史：华为B3-241H显示器的护眼模式怎么开？；本轮：它怎么恢复出厂设置？
- VAL-DIV-002（operation_step）：在H3CMSR-XS系列路由器上配置端口映射，从进入NAT配置页面到点击确定完成，具体步骤是怎样的？
- VAL-DIV-025（product_fact）：H3CMER系列路由器里，抓包工具配置时“抓包长度”这个参数的单位是什么？
- VAL-DIV-001（table_lookup）：H3CMSR-XS系列路由器上，日志级别数值为2对应哪个信息级别、描述是什么？
- VAL-DIV-008（troubleshooting）：打印出来的整页都模糊不清，可能是什么原因，该怎么处理？

## 召回对照：原有与新增分开报告

各组使用同一批审核后的商品范围及独立问题，排除实体识别和指代改写错误；这是给定正确商品的条件检索实验，不能视为Agent端到端成绩。
主指标取融合候选的Recall@20、Precision@20，辅助看重排nDCG@10及标注覆盖率；不以最终上下文事实覆盖代替召回指标。

| 样本范围 | 策略 | 可回答题n | Recall@20 | Precision@20 | 重排nDCG@10 | 前20标注覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | main | 20 | 64.12% | 31.25% | 0.6243 | 95.20% |
| original | main_hyde | 20 | 64.55% | 31.50% | 0.6291 | 95.60% |
| original | main_subquery | 20 | 64.01% | 31.00% | 0.6338 | 94.80% |
| original | main_hyde_subquery | 20 | 65.39% | 32.00% | 0.6247 | 95.00% |
| new | main | 105 | 84.71% | 16.81% | 0.8010 | 97.33% |
| new | main_hyde | 105 | 87.91% | 17.57% | 0.8122 | 97.38% |
| new | main_subquery | 105 | 85.96% | 17.19% | 0.8009 | 97.38% |
| new | main_hyde_subquery | 105 | 87.50% | 17.48% | 0.8091 | 97.47% |
| all | main | 125 | 81.42% | 19.12% | 0.7727 | 96.92% |
| all | main_hyde | 125 | 84.17% | 19.80% | 0.7829 | 97.04% |
| all | main_subquery | 125 | 82.45% | 19.40% | 0.7742 | 96.88% |
| all | main_hyde_subquery | 125 | 83.96% | 19.80% | 0.7796 | 97.00% |

题目结构与标注方式发生变化，不能把新集分数与旧集分数的差值当作系统提升。未标注结果按零收益计，需同时阅读标注覆盖；无答案题不计入Recall/Precision的平均值。
Precision@20固定以20为分母，返回不足20条时空位计零；部分短手册的候选不足20，不能把这种固定K效应误判为噪声增加。

本轮点估计：主查询加入子查询，Recall@20变化+1.03个百分点；已有HyDE时再加子查询，变化-0.21个百分点。这些是相关手册上的合成开发样本，标注也未经人工校准，不能据此宣称稳定的泛化增益。

## 复现

```powershell
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/expand_source_benchmark.py generate --output-dir eval/local/diverse_build_v1 --repair-source-spans --concurrency 3
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/propose_qrels.py --dataset eval/local/diverse_build_v1/draft --selection eval/local/diverse_build_v1/selection.json --output-dir eval/local/diverse_build_v1/proposals --batch-size 8 --concurrency 4
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/audit_qrel_sources.py --dataset eval/local/diverse_build_v1/draft --selection eval/local/diverse_build_v1/selection.json --sources eval/local/source_pdf_index.jsonl --proposals eval/local/diverse_build_v1/proposals --output-dir eval/local/diverse_build_v1/audit --allow-incomplete --concurrency 4
python scripts/expand_source_benchmark.py publish --output-dir eval/local/diverse_build_v1 --publish-to eval/local/productdoc-diverse-v1
python scripts/with_eval_env.py --env-file API_ENV_PATH scripts/eval_subquery_recall.py --dataset eval/local/productdoc-diverse-v1 --reviewed-inputs --output-dir logs/subquery_recall_diverse_v1 --concurrency 3
python scripts/summarize_benchmark_diversity.py --dataset eval/local/productdoc-diverse-v1 --experiment logs/subquery_recall_diverse_v1/retrieval.json --output eval/DIVERSITY_RESULTS.md
```

数据指纹：`7d3813e91ed6908788b73bf5f0134e3170159b31d82e1980adb070229415946b`。原始PDF、真实题目、引用、模型缓存和完整报告保留在本地；仓库提供脚本与汇总。
