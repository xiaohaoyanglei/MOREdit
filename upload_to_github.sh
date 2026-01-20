#!/bin/bash
# Pointer LoRA GitHub 上传脚本

set -e

echo "=== Pointer LoRA GitHub 上传向导 ==="
echo ""

# 检查是否已经是 git 仓库
if [ -d ".git" ]; then
    echo "⚠️  已存在 .git 目录，跳过初始化"
else
    echo "📦 初始化 Git 仓库..."
    git init
fi

# 配置 git 用户信息
echo ""
echo "请配置 Git 用户信息："
read -p "Git 用户名: " git_username
read -p "Git 邮箱: " git_email

git config user.name "$git_username"
git config user.email "$git_email"

echo "✓ Git 配置完成"

# 检查 .gitignore
if [ ! -f ".gitignore" ]; then
    echo "⚠️  未找到 .gitignore 文件，建议先创建"
    exit 1
fi

# 显示将要提交的文件
echo ""
echo "📋 将要提交的文件："
git status

echo ""
read -p "确认添加这些文件？(y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# 添加文件
echo ""
echo "📝 添加文件到 Git..."
git add .

# 创建 commit
read -p "Commit 信息（留空使用默认）: " commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Initial commit: Pointer LoRA for Qwen edit with ControlNet inpainting"
fi

git commit -m "$commit_msg"
echo "✓ Commit 创建完成"

# 配置远程仓库
echo ""
echo "请在 GitHub 上创建一个新的空仓库（不要初始化 README）"
echo "然后输入仓库信息："
read -p "GitHub 用户名: " github_username
read -p "仓库名称: " repo_name

remote_url="https://github.com/$github_username/$repo_name.git"

# 检查是否已有 origin
if git remote | grep -q "^origin$"; then
    echo "⚠️  已存在 origin 远程仓库，更新 URL..."
    git remote set-url origin "$remote_url"
else
    echo "🔗 添加远程仓库..."
    git remote add origin "$remote_url"
fi

# 推送到 GitHub
echo ""
echo "🚀 推送到 GitHub..."
git branch -M main

echo ""
echo "准备推送到: $remote_url"
echo "⚠️  如果使用 HTTPS，需要使用 Personal Access Token 作为密码"
echo "   获取 Token: https://github.com/settings/tokens"
echo ""

read -p "确认推送？(y/n): " push_confirm
if [ "$push_confirm" != "y" ]; then
    echo "已取消推送"
    exit 0
fi

git push -u origin main

echo ""
echo "✅ 成功！仓库已上传到："
echo "   https://github.com/$github_username/$repo_name"
