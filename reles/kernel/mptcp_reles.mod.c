#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(.gnu.linkonce.this_module) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section(__versions) = {
	{ 0x2995df5e, "module_layout" },
	{ 0xcc8440d6, "param_ops_byte" },
	{ 0x7847bf4d, "param_ops_bool" },
	{ 0x461c82a4, "mptcp_unregister_scheduler" },
	{ 0x18c27524, "mptcp_register_scheduler" },
	{ 0xc5850110, "printk" },
	{ 0x3fceed7, "tcp_current_mss" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "C02C7ABFD306838CAEE7521");
