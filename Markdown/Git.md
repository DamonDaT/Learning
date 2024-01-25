### 【说明】

***

Git 常用知识点





### 【一】本地工作区和版本库

***

##### 【1.1】初始化

```shell
git init
```

##### 【1.2】日志（提交记录）

```shell
git log [--graph] [--oneline] [--pretty=oneline] [--abbrev-commit]
```

##### 【1.3】历史（操作记录）

```shell
git reflog
```

##### 【1.4】状态

```shell
git status
```

##### 【1.5】添加到暂存区

```shell
git add [file/re]
```

##### 【1.6】提交到版本库

```shell
git commit -m "[xxx] yyy"
```

##### 【1.7】丢弃工作区修改

```shell
git restore [file/re]
```

##### 【1.8】丢弃暂存区修改（保留工作区）

```shell
git restore --staged [file/re]  /  git reset HEAD [file/re]
```

##### 【1.9】丢弃版本库的修改（保留工作区和暂存区）

```shell
git reset --soft [commit_id/reflog_tag]
```

##### 【1.10】丢弃版本库的修改（丢弃工作区和暂存区）

```shell
git reset --hard [commit_id/reflog_tag]
```

##### 【1.11】还原删除的文件（前提版本库里还存在）

```shell
git checkout -- [file/re]
```

##### 【1.12】删除版本库文件

```shell
git rm [file/re]
```



### 【二】远程仓库

***

##### 【2.1】远程标志

```shell
git remote [-v] [add origin xxx.git] [rm remote_branch]
```

##### 【2.2】克隆

```shell
git clone [xxx.git]
```

##### 【2.3】推送远程仓库

```shell
git push [-u] [remote_branch] [local_branch]
```

##### 【2.4】拉取

```shell
git pull
```



### 【三】分支管理

***

##### 【3.1】查看

```shell
git branch [-a]
```

##### 【3.2】新建

```shell
git branch [local_branch_name]
```

##### 【3.3】删除

```shell
git branch [-d/-D] [local_branch_name]
```

##### 【3.4】建立链接

```shell
git branch --set-upstream-to [local_branch_name] [origin/remote_branch_name]
```

##### 【3.5】切换 / 新建并切换

```shell
git switch [local_branch_name]
```

##### 【3.6】合并（Fast Forward）

```shell
git merge [other_branch_name]
```

##### 【3.7】合并（禁用 Fast Forward）

```shell
git merge --no-ff -m "[xxx] yyy" [other_branch_name]
```

##### 【3.8】远程分支同步到本地

```shell
git fetch --prune
```



### 【四】进阶用法

***

##### 【4.1】缓存

```shell
git stash list
```

##### 【4.2】查看缓存

```shell
git stash list
```

##### 【4.3】恢复缓存（缓存不删除）

```shell
git stash apply [stash_id]
```

##### 【4.4】恢复缓存（恢复并删除）

```shell
git stash pop
```

##### 【4.5】复制提交到当前分支

```shell
git cherry-pick [commit_id]
```

##### 【4.6】合并多次提交（整洁）

```shell
git rebase -i [commit_id] # 涵盖范围是 [最新提交, commit_id) 注意开闭区间，需要配合 <git push -f> 使用，强制推送远端
```



### 【五】标签

***

##### 【5.1】打标签

```shell
git tag [tag_name] [commit_id]
```

##### 【5.2】打标签（添加描述）

```shell
git tag -a [tag_name] -m "xxx" [commit_id]
```

##### 【5.3】查看所有标签

```shell
git tag
```

##### 【5.4】查看标签信息

```shell
git show [tag_name]
```

##### 【5.5】推送本地标签

```shell
git push origin [tag_name/--tags]
```

##### 【5.6】删除本地标签

```shell
git tag -d [tag_name]
```

##### 【5.7】删除远程标签

```shell
git push origin [:refs/tags/tag_name]
```
