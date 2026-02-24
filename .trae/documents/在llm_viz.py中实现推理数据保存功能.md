## 计划内容

### 1. 在llm_viz.py中添加保存推理数据的功能
- 添加必要的导入（json, os, pickle, datetime）
- 实现`save_inference_data`函数，保存以下数据：
  - 图片（input_image.png）
  - 提示词和生成文本（metadata.json）
  - 注意力矩阵（attention_weights.pkl）
  - 输入IDs（input_ids.pkl）
  - 图像网格信息（image_grid_thw.pkl）
  - 生成的tokens（tokens.json）
- 在生成文本后调用保存函数

### 2. 确保数据保存格式一致性
- 使用与之前相同的目录结构（inference_data/时间戳子目录）
- 使用相同的文件命名和格式
- 确保注意力矩阵的存储格式兼容

### 3. 验证visualize_saved_data.py的兼容性
- 确保visualize_saved_data.py能够正确加载llm_viz.py保存的数据
- 如有必要，进行小的调整以确保兼容性

### 4. 测试功能
- 运行llm_viz.py生成并保存数据
- 使用visualize_saved_data.py加载并可视化保存的数据
- 验证所有功能正常工作

## 技术实现细节

- **保存位置**：inference_data/时间戳命名的子目录
- **数据格式**：
  - 图片：PNG格式
  - 元数据：JSON格式
  - 张量数据：pickle格式（转换为numpy数组）
  - Tokens：JSON格式
- **保存时机**：在文本生成完成后立即保存
- **兼容性**：确保与之前的保存格式完全兼容，以便visualize_saved_data.py可以直接使用

## 预期结果

完成后，用户可以：
1. 使用llm_viz.py进行推理，数据会自动保存
2. 使用visualize_saved_data.py直接加载保存的数据进行可视化
3. 无需修改app.py，所有功能都在llm_viz.py中实现