# -*- mode: python -*-

import sys, os
#sys.path.append('D:\\virtual_py_no1\\env1\\Lib\\site-packages\\pywin32_system32\\')
#sys.path.append('D:\\janvorac\\pythons\\py3_moes\\Lib\site-packages\\pypiwin32_system32')
#sys.path.append(os.path.join('D:', 'janvorac', 'pythons', 'py3_moes', 'Lib', 'site-packages', 'pypiwin32_system32'))

block_cipher = None


a = Analysis(['GUI.py'],
             #pathex=['D:/janvorac/massiveoes/massiveOES/'],
             #pathex=[os.path.abspath(''), 
		#os.path.abspath(os.path.join(os.sep, 'D:', 'Python36', 'Lib', 'site-packages', 'pypiwin32_system32'))],
             pathex = ['C:/Users/Petr/Envs/py1/Lib/site-packages/pypiwin32_system32'],
             #pathex = [],
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
                            'scipy.optimize.linesearch',
                            'enable.qt4.image',
                            'scipy._lib.messagestream',
                            'pandas._libs.tslibs.timedeltas',
                            'pkg_resources.resource_string'
                        ],
             hookspath=None,
#             runtime_hooks=['rthook_pyqt4.py'],
             runtime_hooks=None,
             excludes=None,
             win_no_prefer_redirects=None,
             win_private_assemblies=None,
             cipher=block_cipher)



from glob import glob
env_path = 'C:/Users/Petr/Envs/py1/Lib/site-packages'
#env_path = os.path.abspath(os.path.join(os.sep,'D:','Python36','Lib','site-packages'))

#files = glob(env_path+'pyface/ui/qt4/images/*')
#files += glob(env_path + 'pyface/action/images/*')
#files += glob(env_path + 'pyface/dock/images/*')
files = glob(os.path.abspath(os.path.join(env_path, 'pyface', 'images','*')))
a.datas += [(os.path.join('pyface', 'images', os.path.split(f)[-1]), f, 'DATA') for f in files]

files = glob(os.path.abspath(os.path.join(os.sep, 'C:','moje_balicky', 'massiveoes ', 'massiveoes', 'massiveOES', 'data', '*')))
a.datas += [(os.path.join('data', os.path.split(f)[-1]), f, 'DATA') for f in files]

files = glob(os.path.abspath(os.path.join(os.sep, 'C:','moje_balicky', 'massiveoes ', 'massiveoes', 'massiveOES', 'images', '*')))
a.datas += [(os.path.join('images', os.path.split(f)[-1]), f, 'DATA') for f in files]

files = glob(os.path.abspath(os.path.join(os.sep, 'C:','moje_balicky', 'massiveoes ', 'massiveoes', 'massiveOES', 'sample_files', '*')))
a.datas += [(os.path.join('sample_files', os.path.split(f)[-1]), f, 'DATA') for f in files]


#l = len(env_path)

#for f in files:
#    print(f[l:])

#a.datas += [(f[l:], f, 'DATA') for f in files]


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher, name = 'moes')
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
