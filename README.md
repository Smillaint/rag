# rag

## API Key 配置

不要把 API key、token、secret 等敏感信息提交到 Git。仓库已经通过 `.gitignore` 忽略 `.env`、`key.txt`、`*.key` 等本地密钥文件。

运行生成测试前，在本地环境变量中配置 API key：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
python test_generator.py
```

如果 API key 已经上传到 GitHub，请立即在对应平台作废/轮换这个 key。`.gitignore` 只能防止后续继续提交，不能从 GitHub 历史记录中移除已经泄露的密钥。
