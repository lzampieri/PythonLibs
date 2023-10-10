# -*- mode: python -*-

import sys, os

block_cipher = None


a = Analysis(['GUI.py'],
             pathex=['~/packages/massiveOES/massiveOES'],
             binaries=None,
             datas=None,
             hiddenimports=['pyface.qt', 
                            'pyface.qt.QtCore', 
                            'pyface.ui.qt4.init',
                            'pyface.ui.qt4.image_resource',
                            'pyface.ui.qt4.resource_manager',
                            'pyface.qt.QtSvg',
                            'pyface.ui.qt4.action',
                            'pyface.ui.qt4.window',
                            'pyface.ui.qt4.gui',
                            'pyface.ui.qt4.widget',
                            'pyface.ui.qt4.timer',
                            'pyface.ui.qt4.timer.timer',
                            'pyface.ui.qt4.about_dialog',
                            'pyface.ui.qt4.dialog',
                            'pyface.ui.qt4.clipboard',
                            'pyface.ui.qt4.action.action_item',
                            'pyface.ui.qt4.action.menu_manager',
                            'pyface.ui.qt4.action.menu_bar_manager',
                            'pyface.ui.qt4.action.status_bar_manager',
                            'pyface.ui.qt4.action.tool_bar_manager',
                            'pyface.ui.qt4.image_cache',
                            'pyface.ui.qt4.timer.do_later',
                            'pyface.ui.qt4.application_window',
                            'pyface.ui.qt4.beep',
                            'pyface.ui.qt4.confirmation_dialog',
                            'pyface.ui.qt4.directory_dialog',
                            'pyface.ui.qt4.file_dialog',
                            'pyface.ui.qt4.heading_text',
                            'pyface.ui.qt4.message_dialog',
                            'pyface.ui.qt4.progress_dialog',
                            'pyface.ui.qt4.python_editor',
                            'pyface.ui.qt4.code_editor',
                            'pyface.ui.qt4.code_editor.code_widget',
                            'pyface.ui.qt4.code_editor.find_widget',
                            'pyface.ui.qt4.code_editor.gutters',
                            'pyface.ui.qt4.code_editor.replace_widget',
                            'pyface.ui.qt4.code_editor.pygments_highlighter',
                            'pyface.ui.qt4.python_shell',
                            'pyface.ui.qt4.console',
                            'pyface.ui.qt4.console.api',
                            'pyface.ui.qt4.console.bracket_matcher',
                            'pyface.ui.qt4.console.call_tip_widget',
                            'pyface.ui.qt4.console.completion_lexer',
                            'pyface.ui.qt4.console.console_widget',
                            'pyface.ui.qt4.console.history_console_widget',
                            'pyface.ui.qt4.splash_screen',
                            'pyface.ui.qt4.split_widget',
                            'pyface.ui.qt4.system_metrics',
                            'pyface.qt.QtGui',
                            'pyface.ui.qt4', 
                            'traitsui.qt4',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'enable.qt4.image',
                        ],
             hookspath=None,
             runtime_hooks=None,
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)

from glob import glob
env_path ='/home/janvorac/.virtualenvs/py3_moes/lib/python3.5/site-packages'

#files = glob(env_path+'pyface/ui/qt4/images/*')
#files += glob(env_path + 'pyface/action/images/*')
#files += glob(env_path + 'pyface/dock/images/*')
files = glob(os.path.join(env_path,'pyface','images','*'))
a.datas += [(os.path.join('pyface','images',os.path.split(f)[-1]), f, 'data') for f in files]

files = glob(os.path.abspath(os.path.join('data', '*')))
a.datas += [(os.path.join('data', os.path.split(f)[-1]), f, 'DATA') for f in files]

files = glob(os.path.abspath(os.path.join('images', '*')))
a.datas += [(os.path.join('images', os.path.split(f)[-1]), f, 'DATA') for f in files]

files = glob(os.path.abspath(os.path.join('sample_files', '*')))
a.datas += [(os.path.join('sample_files', os.path.split(f)[-1]), f, 'DATA') for f in files]



### needed for pandas
def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path
dict_tree = Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc"])
a.datas += dict_tree
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)
###


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='massiveOES',
          debug=False,
          strip=None,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='massiveOES')
