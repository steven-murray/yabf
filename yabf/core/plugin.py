"""
Provides common plugin framework
"""


def plugin_mount_factory():
    """
    http://martyalchin.com/2008/jan/10/simple-plugin-framework/

    Returns
    -------

    """
    class PluginMount(type):
        def __init__(cls, name, bases, attrs):
            if not hasattr(cls, '_plugins'):
                cls._plugins = {}
            else:
                cls._plugins[cls.__name__] = cls

        def _get_plugin(cls, *args, **kwargs):
            return cls._plugins[cls.__name__](*args, **kwargs)

    return PluginMount

