# -*- coding: utf-8 -*-
# トピックモデル(latent dirichlet allocation)
import sys
import math
import random
import argparse
import scipy.special
from collections import defaultdict


class LDA:
    def __init__(self, data):
        self.corpus_file = data
        self.target_word = defaultdict(int)
        self.corpus = []
        comment = ""
        for strm in open(data, "r"):
            document = {}
            if strm.startswith("#"):
                comment = strm.strip()
            else:
                if comment:
                    document["comment"] = comment
                words = strm.strip().split(" ")
                document["bag_of_words"] = words
                for v in words:
                    self.target_word[v] += 1
                self.corpus.append(document)
        self.V = float(len(self.target_word))
        self.z_d_n = []                                                     # d番目の文書のn番目の文字のトピック
        self.m_d_z = defaultdict(lambda: defaultdict(lambda: self.alpha))   # d番目の文書でトピックzが割り当てられた単語の数
        self.m_z_v = defaultdict(lambda: defaultdict(lambda: self.beta))    # トピックzに割り当てられた語彙vの数
        self.m_z = defaultdict(lambda: self.beta * self.V)                  # データ全体でトピックzに割り当てられた単語の数

    def set_param(self, alpha, beta, K, N, converge):
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.N = N
        self.converge = converge

    def likelihood(self):
        phi = defaultdict(lambda:defaultdict(float))
        for z in xrange(self.K):
            for v in self.target_word:
                phi[z][v] = self.m_z_v[z][v] / self.m_z[z]
        likelihoods = []
        for d, doc in enumerate(self.corpus):
            likelihood = 0.0
            theta = defaultdict(float)
            for z in xrange(self.K):
                theta[z] = self.m_d_z[d][z] / (len(doc["bag_of_words"]) + self.K * self.alpha)
            for v in doc["bag_of_words"]:
                inner = 0
                for z in xrange(self.K):
                    inner += phi[z][v] * theta[z]
                likelihood += math.log(inner)
            likelihoods.append(likelihood)
        return sum(likelihoods)/len(likelihoods)

    def initialize(self):
        for d, doc in enumerate(self.corpus):
            z_n = []    # n番目の単語のトピックz
            for v in doc["bag_of_words"]:
                z = random.randint(0, self.K-1)
                z_n.append(z)
                self.m_d_z[d][z] += 1
                self.m_z_v[z][v] += 1
                self.m_z[z] += 1
            self.z_d_n.append(z_n[::])

    def learn(self):
        self.initialize()
        self.lkhds = []
        for i in xrange(self.N):
            self.gibbs_sampling()
            sys.stderr.write("iteration=%d/%d K=%s alpha=%s beta=%s\n"%(i+1, self.N, self.K, self.alpha, self.beta))
            if i % 10 == 0:
                self.n = i+1
                self.lkhds.append(self.likelihood())
                sys.stderr.write("%s : likelihood=%f\n"%(i+1, self.lkhds[-1]))
                if len(self.lkhds) > 1:
                    diff = self.lkhds[-1] - self.lkhds[-2]
                    if math.fabs(diff) < self.converge:
                        break
        self.n = i+1

    def gibbs_sampling(self):
        for d, doc in enumerate(self.corpus):
            for n, v in enumerate(doc["bag_of_words"]):
                self.sample_word(d, n, v)       # d番目の文書のn番目の単語vのトピックをサンプリング
        # ハイパーパラメータalphaの更新
        numerator = 0.0
        denominator = 0.0
        for d, doc in enumerate(self.corpus):
            for z in xrange(1, self.K + 1):
                numerator += (scipy.special.digamma(self.m_d_z[d][z] + self.alpha))
            denominator += scipy.special.digamma(len(self.corpus[d]["bag_of_words"]) + (self.alpha * self.K))
        numerator -= (len(self.corpus) * self.K * scipy.special.digamma(self.alpha))
        denominator = self.K * denominator
        denominator -= (len(self.corpus) * self.K * scipy.special.digamma(self.alpha * self.K))
        self.alpha = self.alpha * (numerator / denominator)
        # ハイパーパラメータbetaの更新
        numerator = 0.0
        denominator = 0.0
        for z in xrange(1, self.K + 1):
            for v in self.target_word.iterkeys():
                numerator += scipy.special.digamma(self.m_z_v[z][v] + self.beta)
            denominator += scipy.special.digamma(self.m_z[z] + (self.beta * self.V))
        numerator -= (self.K * self.V * scipy.special.digamma(self.beta))
        denominator = ((self.V * denominator) - (self.K * self.V * scipy.special.digamma(self.beta * self.V)))
        self.beta = self.beta * (numerator / denominator)

    def sample_word(self, d, n, v):
        z = self.z_d_n[d][n]                    # Step1: 現在のd番目の文書のn番目の単語に割り当てられているトピックzについてのカウントを減らす
        self.m_d_z[d][z] -= 1
        self.m_z_v[z][v] -= 1
        self.m_z[z] -= 1
        p_z = defaultdict(float)                # Step2: 可能なトピックの分布p_zをディリクレ過程の式で計算 (嘘。ディリクレ過程じゃなかった。調査が必要)
        for z in xrange(self.K):
            p_z[z] = self.m_z_v[z][v] * self.m_d_z[d][z] / self.m_z[z]
        new_z = self.sample_one(p_z)            # Step3: Step2で計算した確率分布に従い新しいトピックをサンプリング
        self.z_d_n[d][n] = new_z                # Step4: Step3でサンプルした新しいトピックを追加する
        self.m_d_z[d][new_z] += 1
        self.m_z_v[new_z][v] += 1
        self.m_z[new_z] += 1

    def sample_one(self, prob_dict):
        z = sum(prob_dict.values())                     # 確率の和を計算
        remaining = random.uniform(0, z)                # [0, z)の一様分布に従って乱数を生成
        for state, prob in prob_dict.iteritems():       # 可能な確率を全て考慮(状態数でイテレーション)
            remaining -= prob                           # 現在の仮説の確率を引く
            if remaining < 0.0:                         # ゼロより小さくなったなら，サンプルのIDを返す
                return state

    def output_model(self):
        print "model\tlda"
        print "@parameter"
        print "corpus_file\t%s"%self.corpus_file
        print "hyper_parameter_alpha\t%f"%self.alpha
        print "hyper_parameter_beta\t%f"%self.beta
        print "number_of_topic\t%d"%self.K
        print "number_of_iteration\t%d"%self.N
        print "@likelihood"
        print "initial likelihood\t%s"%(self.lkhds[0])
        print "last likelihood\t%s"%(self.lkhds[-1])
        print "@vocaburary"
        for v in self.target_word:
            print "target_word\t%s"%v
        print "@count"
        for z, dist in self.m_z_v.iteritems():
            print "m_z\t%s\t%s"%(z, int(self.m_z[z] - (self.beta * self.V)))
            for v, freq in sorted(dist.iteritems(), key=lambda x:x[1], reverse=True):
                freq = int(freq - self.beta)
                if freq > 0:
                    print "m_z_v\t%s\t%s\t%s"%(z, v, freq)

def main(args):
    lda = LDA(args.data)
    lda.set_param(args.alpha, args.beta, args.K, args.N, args.converge)
    lda.learn()
    lda.output_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", dest="alpha", default=0.01, type=float, help="hyper parameter alpha")
    parser.add_argument("-b", "--beta", dest="beta", default=0.01, type=float, help="hyper parameter beta")
    parser.add_argument("-k", "--K", dest="K", default=10, type=int, help="topic")
    parser.add_argument("-n", "--N", dest="N", default=1000, type=int, help="max iteration")
    parser.add_argument("-c", "--converge", dest="converge", default=0.01, type=str, help="converge")
    parser.add_argument("-d", "--data", dest="data", default="data.txt", type=str, help="training data")
    args = parser.parse_args()
    main(args)
