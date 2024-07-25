import numpy as np
import numpy.random as nr


#共通後処理
"""
pool=(shots, N), score=(N, )
"""
def get_result(pool, score, index_map):
    #重複解を集計
    unique_pool, original_index, unique_counts = np.unique(pool, axis=0, return_index=True, return_counts=True)
    #print(unique_pool, original_index, unique_counts)
    
    #エネルギーもユニークに集計
    unique_energy = score[original_index]
    #print(unique_energy)
    
    #エネルギー低い順にソート
    order = np.argsort(unique_energy)
    unique_pool = unique_pool[order]
    unique_energy = unique_energy[order]
    unique_counts = unique_counts[order]
    
    #結果リスト
    result = [[dict(zip(index_map.keys(), unique_pool[i])), unique_energy[i], unique_counts[i]] for i in range(len(unique_pool))]
    #print(result)
    
    return result


#アニーリング
class SASampler:
    def __init__(self, seed=None):
        #乱数シード
        self.seed = seed

    
    def run(self, hobo, shots=100, T_num=2000, show=False):
        
        #matrixサイズ
        N = len(hobo)
        # print(N)
        
        #次数
        ho = len(hobo.shape)
        # print(ho)
        
        #シード固定
        nr.seed(self.seed)
        
        #
        shots = max(int(shots), 100)
        
        # プール初期化
        pool_num = shots
        pool = nr.randint(0, 2, (pool_num, N)).astype(float)
        # print(pool)
        
        #スコア初期化
        score = np.zeros(pool_num)
        # score2 = np.zeros(pool_num)
        
        #スコア計算コマンド
        k = 'a,b,c,d,e,f,g,h,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'
        l = 'abcdefghjklmnopqrstuvwxyz'
        s = k[:2*ho] + l[:ho]
        # print(s)
        command = 'global score2\r\n'
        command += f'score2 = np.zeros(pool_num)\r\n'
        command += f'for i in range(pool_num):\r\n'
        command += f'    score2[i] = np.einsum(\'{s}\', pool2[i]' + ', pool2[i]' * (ho - 1) + f', hobo)\r\n'
        # print(command)
        
        #スコア計算
        pool2 = pool
        exec(command)
        score = score2
        # print(score)

        # フリップ数リスト（2個まで下がる）
        flip = np.sort(nr.rand(T_num) ** 2)[::-1]
        flip = (flip * max(0, N * 0.5 - 2)).astype(int) + 2
        # print(flip)
        
        # フリップマスクリスト
        flip_mask = [[1] * flip[0] + [0] * (N - flip[0])]
        if N <= 2:
            flip_mask = np.ones((T_num, N), int)
        else:
            for i in range(1, T_num):
                tmp = [1] * flip[i] + [0] * (N - flip[i])
                nr.shuffle(tmp)
                # 前と重複なら振り直し
                while tmp == flip_mask[-1]:
                    nr.shuffle(tmp)
                flip_mask.append(tmp)
            flip_mask = np.array(flip_mask, bool)
        # print(flip_mask.shape)
        
        # 局所探索フリップマスクリスト
        single_flip_mask = np.eye(N, dtype=bool)
        
        """
        アニーリング＋1フリップ
        """
        # アニーリング
        # 集団まるごと温度を下げる
        for fm in flip_mask:
            # フリップ後　pool_num, N
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            # print(command)
            exec(command)
            # print(score)
            # print(score2)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        
        # 最後に1フリップ局所探索
        # 集団まるごと
        for fm in single_flip_mask:
            # フリップ後
            # pool2 = np.where(fm, 1 - pool, pool)
            pool2 = pool.copy()
            pool2[:, fm] = 1. - pool[:, fm]
            # score2 = np.sum((pool2 @ qmatrix) * pool2, axis=1)
            
            # print(command)
            exec(command)
            # print(score)
            # print(score2)
            
            # 更新マスク
            update_mask = score2 < score
            # print(update_mask)
    
            # 更新
            pool[update_mask] = pool2[update_mask]
            score[update_mask] = score2[update_mask]
        pool = pool.astype(int)
        
        # ----------
        #共通後処理
        index_map = {f'q{k}': k for k in range(N)}
        result = get_result(pool, score, index_map)
        
        return result



if __name__ == "__main__":
    pass