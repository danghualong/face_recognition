import requests  # http客户端
import re  # 正则表达式模块
import random  # 随机数
import os  # 创建文件夹


def mkdir(path):  # 创建文件夹
    is_exists = os.path.exists(path)
    if not is_exists:
        print('创建名字叫做', path, '的文件夹')
        os.makedirs(path)
        print('创建成功！')
    else:
        print(path, '文件夹已经存在了，不再创建')


def getPic(html, keyword, path):
    print("正在查找：" + keyword + ' 对应的图片，正在从百度图库重下载：  ')
    for addr in re.findall(str('"objURL":"(.*?)"'), html, re.S):
        print("现在正在爬取的URL地址：" + addr)
        try:
            pics = requests.get(addr, timeout=10)
        except requests.exceptions.ConnectionError:
            print("当前Url请求错误")
            continue
        # 假设产生的随机数不重复
        fq = open(path + '//' + str(random.randrange(1000, 2000)) + '.jpg', 'w+b')
        fq.write(pics.content)
        fq.close()
        print('写入完成')


if __name__ == "__main__":
    word = input("请输入关键词：")
    result = requests.get(
        "http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=" + word)
    # print(result.text)
    print("写入完毕")
    folderName = input("请输入文件夹名：")
    path = 'pic/{0}'.format(folderName)  # 保存图片文件夹名称
    mkdir(path)
    getPic(result.text, word, path)
