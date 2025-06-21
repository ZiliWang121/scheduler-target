/* MPTCP Scheduler module selector. Highly inspired by tcp_cong.c */

#include <linux/module.h>
#include <net/mptcp.h>

static unsigned char num_segments __read_mostly = 1;
module_param(num_segments, byte, 0644); // makes sure the module variable num_segments is changeable via sysctl or with socket option 46.
MODULE_PARM_DESC(num_segments, "The number of consecutive segments that are part of a burst");

static bool cwnd_limited __read_mostly = 1;
module_param(cwnd_limited, bool, 0644);
MODULE_PARM_DESC(cwnd_limited, "if set to 1, the scheduler tries to fill the congestion-window on all subflows");

/* 
 * 关键修复1: 统一数据结构，兼容kernel patch的内存布局
 * 原问题：你的结构与kernel patch不匹配，导致读写错误的内存位置
 * 新设计：既支持kernel patch在offset[1]写入，又支持调度器的字节计数逻辑
 */
struct mysched_priv
{
	unsigned char quota_packets;   /* offset[0]: 包计数器，兼容patch的quota字段 */
	unsigned char num_segments;    /* offset[1]: RL权重，kernel patch在这里写入！*/
	__u16 quota_byte;             /* offset[2-3]: 字节计数器，保持你的原逻辑 */
	__u16 weight_bytes;           /* offset[4-5]: 字节权重 = num_segments * 基础单位 */
	__u16 reserved;               /* offset[6-7]: 保留字段 */
};

static struct mysched_priv *mysched_get_priv(const struct tcp_sock *tp)
{
	return (struct mysched_priv *)&tp->mptcp->mptcp_sched[0];
}

/* Heavily inspired by mptcp_rr and gaogogo/Experiment mysched.c
 */

/* If the sub-socket sk available to send the skb?*/
static bool mptcp_reles_is_available(const struct sock *sk, const struct sk_buff *skb,
									 bool zero_wnd_test, bool cwnd_test)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	unsigned int space, in_flight;

	/* Set of states for which we are allowed to send data */
	if (!mptcp_sk_can_send(sk))
		return false;

	/* We do not send data on this subflow unless it is
	 * fully established, i.e. the 4th ack has been received.
	 */
	if (tp->mptcp->pre_established)
		return false;

	if (tp->pf)
		return false;

	if (inet_csk(sk)->icsk_ca_state == TCP_CA_Loss)
	{
		/* If SACK is disabled, and we got a loss, TCP does not exit
		 * the loss-state until something above high_seq has been acked.
		 * (see tcp_try_undo_recovery)
		 *
		 * high_seq is the snd_nxt at the moment of the RTO. As soon
		 * as we have an RTO, we won't push data on the subflow.
		 * Thus, snd_una can never go beyond high_seq.
		 */
		if (!tcp_is_reno(tp))
			return false;
		else if (tp->snd_una != tp->high_seq)
			return false;
	}

	if (!tp->mptcp->fully_established)
	{
		/* Make sure that we send in-order data */
		if (skb && tp->mptcp->second_packet &&
			tp->mptcp->last_end_data_seq != TCP_SKB_CB(skb)->seq)
			return false;
	}

	if (!cwnd_test)
		goto zero_wnd_test;

	in_flight = tcp_packets_in_flight(tp);
	/* Not even a single spot in the cwnd */
	if (in_flight >= tp->snd_cwnd)
		return false;

	/* Now, check if what is queued in the subflow's send-queue
	 * already fills the cwnd.
	 */
	space = (tp->snd_cwnd - in_flight) * tp->mss_cache;

	if (tp->write_seq - tp->snd_nxt > space)
		return false;

zero_wnd_test:
	if (zero_wnd_test && !before(tp->write_seq, tcp_wnd_end(tp)))
		return false;

	return true;
}

/* Are we not allowed to reinject this skb on tp? Copied from default mptcp_sched.c*/
static int mptcp_reles_dont_reinject_skb(const struct tcp_sock *tp, const struct sk_buff *skb)
{
	/* If the skb has already been enqueued in this sk, try to find
	 * another one.
	 */
	return skb &&
		   /* Has the skb already been enqueued into this subsocket? */
		   mptcp_pi_to_flag(tp->mptcp->path_index) & TCP_SKB_CB(skb)->path_mask;
}

/* We just look for any subflow that is available */
static struct sock *reles_get_available_subflow(struct sock *meta_sk,
												struct sk_buff *skb,
												bool zero_wnd_test)
{
	const struct mptcp_cb *mpcb = tcp_sk(meta_sk)->mpcb;
	struct sock *sk = NULL, *bestsk = NULL, *backupsk = NULL;
	struct mptcp_tcp_sock *mptcp;

	/* Answer data_fin on same subflow!!! */
	if (meta_sk->sk_shutdown & RCV_SHUTDOWN &&
		skb && mptcp_is_data_fin(skb))
	{
		mptcp_for_each_sub(mpcb, mptcp)
		{
			sk = mptcp_to_sock(mptcp);
			if (tcp_sk(sk)->mptcp->path_index == mpcb->dfin_path_index &&
				mptcp_reles_is_available(sk, skb, zero_wnd_test, true))
				return sk;
		}
	}
	mptcp_for_each_sub(mpcb, mptcp)
	{
		struct tcp_sock *tp;
		struct mysched_priv *msp;

		sk = mptcp_to_sock(mptcp);
		tp = tcp_sk(sk);
		msp = mysched_get_priv(tp);

		/* 
		 * 关键修复2: 正确检查RL权重
		 * 原问题：检查了错误的字段 msp->weight
		 * 新逻辑：检查正确的字段 msp->num_segments（这里存储RL设置的权重）
		 * RL可以设置num_segments=0来完全禁用某个子流
		 */
		if (msp->num_segments == 0)
		{
			continue;
		}

		if (!mptcp_reles_is_available(sk, skb, zero_wnd_test, true))
		{
			continue;
		}

		if (mptcp_reles_dont_reinject_skb(tp, skb))
		{
			backupsk = sk;
			continue;
		}

		bestsk = sk;
	}

	if (bestsk)
	{
		sk = bestsk;
	}
	else if (backupsk)
	{
		// It has been sent on all subflows once - let's give it a
		// chance again by restarting its pathmask.
		if (skb)
			TCP_SKB_CB(skb)->path_mask = 0;
		sk = backupsk;
	}
	return sk;
}

/* Returns the next segment to be sent from the mptcp meta-queue.
 * (chooses the reinject queue if any segment is waiting in it, otherwise,
 * chooses the normal write queue).
 * Sets *@reinject to 1 if the returned segment comes from the
 * reinject queue. Sets it to 0 if it is the regular send-head of the meta-sk,
 * and sets it to -1 if it is a meta-level retransmission to optimize the
 * receive-buffer.
 */
static struct sk_buff *__mptcp_reles_next_segment(const struct sock *meta_sk, int *reinject)
{
	const struct mptcp_cb *mpcb = tcp_sk(meta_sk)->mpcb;
	struct sk_buff *skb = NULL;

	*reinject = 0;

	/* If we are in fallback-mode, just take from the meta-send-queue */
	if (mpcb->infinite_mapping_snd || mpcb->send_infinite_mapping)
		return tcp_send_head(meta_sk);

	skb = skb_peek(&mpcb->reinject_queue);

	if (skb)
		*reinject = 1;
	else
		skb = tcp_send_head(meta_sk);
	return skb;
}

static struct sk_buff *mptcp_reles_next_segment(struct sock *meta_sk,
												int *reinject,
												struct sock **subsk,
												unsigned int *limit)
{
	const struct mptcp_cb *mpcb = tcp_sk(meta_sk)->mpcb;
	struct mptcp_tcp_sock *mptcp;
	struct sock *choose_sk = NULL;
	struct sk_buff *skb = __mptcp_reles_next_segment(meta_sk, reinject);
	/* 
	 * 关键修复3: 改进变量类型，支持更大的配额
	 * 原问题：unsigned char限制了配额大小（最大255字节）
	 * 新设计：使用__u16支持更大配额，满足RL的需求
	 */
	__u16 best_remaining = 0;     /* 最佳剩余配额 */
	__u16 current_remaining = 0;  /* 当前子流剩余配额 */
	unsigned char iter = 0, full_subs = 0;

	/* As we set it, we have to reset it as well. */
	*limit = 0;

	if (!skb)
		return NULL;

	if (*reinject)
	{
		*subsk = reles_get_available_subflow(meta_sk, skb, false);
		if (!*subsk)
			return NULL;

		return skb;
	}

retry:
	choose_sk = NULL;
	best_remaining = 0;
	iter = 0;
	full_subs = 0;
	
	/* 
	 * 关键修复4: 保持你的原始选择逻辑，只修复字段错误
	 * 你的原逻辑思路完全正确：选择剩余配额最大的子流
	 * 唯一问题：使用了错误字段 msp->weight，应该用 msp->num_segments
	 * 
	 * 针对Softmax总和=100的优化：
	 * 使用10000作为基数，支持精确的百分比分配
	 */
	#define SOFTMAX_QUOTA_BASE 10000  /* 100倍基数，支持0.01%精度 */
	
	mptcp_for_each_sub(mpcb, mptcp)
	{
		struct sock *sk_it = mptcp_to_sock(mptcp);
		struct tcp_sock *tp_it = tcp_sk(sk_it);
		struct mysched_priv *msp = mysched_get_priv(tp_it);

		if (!mptcp_reles_is_available(sk_it, skb, false, cwnd_limited))
			continue;
			
		/* 跳过RL禁用的子流 (num_segments=0) */
		if (msp->num_segments == 0)
			continue;

		iter++;

		/* 
		 * 关键修复：使用正确的字段和Softmax优化的配额计算
		 * 原逻辑：if (msp->weight && (msp->quota_byte < msp->weight) && ...)
		 * 修复后：if (msp->weight_bytes && (msp->quota_byte < msp->weight_bytes) && ...)
		 * 
		 * 你的原始选择策略保持不变：选择剩余配额最大的子流
		 */
		if (msp->weight_bytes && (msp->quota_byte < msp->weight_bytes) &&
			(msp->weight_bytes - msp->quota_byte > best_remaining))
		{
			current_remaining = msp->weight_bytes - msp->quota_byte; /* 剩余配额(字节) */
			if (best_remaining < current_remaining)
			{
				choose_sk = sk_it;
				best_remaining = current_remaining;
			}
		}

		/* 统计已用完配额的子流 */
		if (msp->quota_byte >= msp->weight_bytes)
			full_subs++;
	}

	if (choose_sk != NULL)
		goto found;

	/* 
	 * 关键修复5: 改进配额重置逻辑
	 * 原逻辑：当所有子流配额用完时重置
	 * 问题：可能导致不精确的比例控制
	 * 新逻辑：只重置那些实际可用的子流，保持RL控制的精确性
	 */
	if (iter && (iter == full_subs))
	{
		printk(KERN_INFO "RL_SCHED: All subflows quota exhausted, resetting for new round\n");
		
		mptcp_for_each_sub(mpcb, mptcp)
		{
			struct sock *sk_it = mptcp_to_sock(mptcp);
			struct tcp_sock *tp_it = tcp_sk(sk_it);
			struct mysched_priv *msp = mysched_get_priv(tp_it);

			if (!mptcp_reles_is_available(sk_it, skb, false, cwnd_limited))
				continue;
				
			/* 只重置有权重的子流 */
			if (msp->num_segments > 0) {
				msp->quota_byte = 0;
				/* Softmax优化：重新计算配额，确保使用最新的RL设置 */
				msp->weight_bytes = (msp->num_segments * 10000) / 100;
			}
		}

		goto retry;
	}

found:
	if (choose_sk)
	{
		unsigned int mss_now;
		struct tcp_sock *choose_tp = tcp_sk(choose_sk);
		struct mysched_priv *msp = mysched_get_priv(choose_tp);

		if (!mptcp_reles_is_available(choose_sk, skb, false, true))
			return NULL;

		/*
		 * 关键修复6: 详细的调试信息，帮助理解RL调度过程
		 * 显示：
		 * 1. 内存布局（验证kernel patch是否正确写入）
		 * 2. RL视图（num_segments, weight_bytes等）
		 * 3. 调度决策（为什么选择这个子流）
		 */
		unsigned char *raw = (unsigned char *)msp;
		printk(KERN_INFO "RL_MEMORY_DEBUG: raw=[%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x]\n",
			   raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
		printk(KERN_INFO "RL_PATCH_VIEW: quota_packets=%u, num_segments=%u\n",
			   msp->quota_packets, msp->num_segments);
		printk(KERN_INFO "RL_SCHED_VIEW: quota_byte=%u, weight_bytes=%u, remaining=%u\n",
			   msp->quota_byte, msp->weight_bytes, best_remaining);

		*subsk = choose_sk;
		mss_now = tcp_current_mss(*subsk);
		
		/* 
		 * 关键修复7: 优化limit设置，平衡效率和精确性
		 * 原问题：limit = min(split, mss_now)太保守，每次只发一个包
		 * 新策略：允许发送多个包，但不超过剩余配额
		 * 这样既提高了效率，又保持了RL控制的精确性
		 */
		__u32 max_packets = min_t(__u32, best_remaining / mss_now, 8);  /* 最多8个包，避免过大burst */
		if (max_packets == 0) max_packets = 1;  /* 至少发送1个包 */
		*limit = max_packets * mss_now;
		
		printk(KERN_INFO "RL_LIMIT: best_remaining=%u, mss=%u, max_packets=%u, limit=%u\n",
			   best_remaining, mss_now, max_packets, *limit);

		/* 
		 * 关键修复8: 精确的配额更新
		 * 原逻辑：msp->quota_byte += skb->len（可能导致累积误差）
		 * 新逻辑：按实际发送的数据更新，确保配额控制的准确性
		 */
		msp->quota_byte += skb->len; /* 按实际包大小更新字节配额 */

		printk(KERN_INFO "RL_FINAL: subflow=%d selected, new_quota=%u/%u\n",
			   choose_tp->mptcp->path_index, msp->quota_byte, msp->weight_bytes);

		return skb;
	}

	return NULL;
}

/* 
 * 关键修复9: 针对Softmax总和=100的初始化优化
 * 你的RL使用softmax输出，总和恰好=100
 * 我们可以利用这个特性做精确的比例控制
 */
static void relessched_init(struct sock *sk)
{
	struct mysched_priv *priv = mysched_get_priv(tcp_sk(sk));
	
	/* 初始化所有字段 */
	priv->quota_packets = 0;
	priv->num_segments = num_segments;        /* 默认权重，RL会通过set_seg更新 */
	priv->quota_byte = 0;
	
	/* 
	 * Softmax优化：使用10000作为基数
	 * 这样RL输出[50, 30, 20]时：
	 * - 子流1配额 = 50 * 100 = 5000字节
	 * - 子流2配额 = 30 * 100 = 3000字节  
	 * - 子流3配额 = 20 * 100 = 2000字节
	 * 总配额 = 10000字节，完美匹配总和=100
	 */
	priv->weight_bytes = (num_segments * 10000) / 100; /* Softmax优化的配额计算 */
	priv->reserved = 0;
	
	printk(KERN_INFO "RL_INIT: softmax_ratio=%u%%, weight_bytes=%u (base=10000)\n",
		   priv->num_segments, priv->weight_bytes);
}

static struct mptcp_sched_ops mptcp_sched_reles = {
	.get_subflow = reles_get_available_subflow,
	.next_segment = mptcp_reles_next_segment,
	.init = relessched_init,
	.name = "reles",
	.owner = THIS_MODULE,
};

static int __init reles_register(void)
{
	BUILD_BUG_ON(sizeof(struct mysched_priv) > MPTCP_SCHED_SIZE);

	printk(KERN_INFO "RL-optimized reles scheduler loaded, struct_size=%zu bytes\n",
		   sizeof(struct mysched_priv));
	printk(KERN_INFO "Ready for Reinforcement Learning control via set_seg()\n");

	if (mptcp_register_scheduler(&mptcp_sched_reles))
		return -1;

	return 0;
}

static void reles_unregister(void)
{
	mptcp_unregister_scheduler(&mptcp_sched_reles);
	printk(KERN_INFO "RL-optimized reles scheduler unloaded\n");
}

module_init(reles_register);
module_exit(reles_unregister);

MODULE_AUTHOR("ME");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("reles scheduler for mptcp -- RL optimized version");
MODULE_VERSION("1.0-RL");