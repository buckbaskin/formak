:py:mod:`formak.ast_tools`
==========================

.. py:module:: formak.ast_tools

.. autodoc2-docstring:: formak.ast_tools
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CompileState <formak.ast_tools.CompileState>`
     - .. autodoc2-docstring:: formak.ast_tools.CompileState
          :summary:
   * - :py:obj:`BaseAst <formak.ast_tools.BaseAst>`
     - .. autodoc2-docstring:: formak.ast_tools.BaseAst
          :summary:
   * - :py:obj:`Public <formak.ast_tools.Public>`
     - .. autodoc2-docstring:: formak.ast_tools.Public
          :summary:
   * - :py:obj:`Private <formak.ast_tools.Private>`
     - .. autodoc2-docstring:: formak.ast_tools.Private
          :summary:
   * - :py:obj:`Arg <formak.ast_tools.Arg>`
     - .. autodoc2-docstring:: formak.ast_tools.Arg
          :summary:
   * - :py:obj:`Namespace <formak.ast_tools.Namespace>`
     - .. autodoc2-docstring:: formak.ast_tools.Namespace
          :summary:
   * - :py:obj:`HeaderFile <formak.ast_tools.HeaderFile>`
     - .. autodoc2-docstring:: formak.ast_tools.HeaderFile
          :summary:
   * - :py:obj:`SourceFile <formak.ast_tools.SourceFile>`
     - .. autodoc2-docstring:: formak.ast_tools.SourceFile
          :summary:
   * - :py:obj:`ClassDef <formak.ast_tools.ClassDef>`
     - .. autodoc2-docstring:: formak.ast_tools.ClassDef
          :summary:
   * - :py:obj:`EnumClassDef <formak.ast_tools.EnumClassDef>`
     - .. autodoc2-docstring:: formak.ast_tools.EnumClassDef
          :summary:
   * - :py:obj:`ForwardClassDeclaration <formak.ast_tools.ForwardClassDeclaration>`
     - .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration
          :summary:
   * - :py:obj:`MemberDeclaration <formak.ast_tools.MemberDeclaration>`
     - .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration
          :summary:
   * - :py:obj:`UsingDeclaration <formak.ast_tools.UsingDeclaration>`
     - .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration
          :summary:
   * - :py:obj:`ConstructorDeclaration <formak.ast_tools.ConstructorDeclaration>`
     - .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration
          :summary:
   * - :py:obj:`ConstructorDefinition <formak.ast_tools.ConstructorDefinition>`
     - .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition
          :summary:
   * - :py:obj:`FunctionDef <formak.ast_tools.FunctionDef>`
     - .. autodoc2-docstring:: formak.ast_tools.FunctionDef
          :summary:
   * - :py:obj:`FunctionDeclaration <formak.ast_tools.FunctionDeclaration>`
     - .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration
          :summary:
   * - :py:obj:`Return <formak.ast_tools.Return>`
     - .. autodoc2-docstring:: formak.ast_tools.Return
          :summary:
   * - :py:obj:`If <formak.ast_tools.If>`
     - .. autodoc2-docstring:: formak.ast_tools.If
          :summary:
   * - :py:obj:`Templated <formak.ast_tools.Templated>`
     - .. autodoc2-docstring:: formak.ast_tools.Templated
          :summary:
   * - :py:obj:`FromFileTemplate <formak.ast_tools.FromFileTemplate>`
     - .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate
          :summary:
   * - :py:obj:`Escape <formak.ast_tools.Escape>`
     - .. autodoc2-docstring:: formak.ast_tools.Escape
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`autoindent <formak.ast_tools.autoindent>`
     - .. autodoc2-docstring:: formak.ast_tools.autoindent
          :summary:

API
~~~

.. py:class:: CompileState
   :canonical: formak.ast_tools.CompileState

   .. autodoc2-docstring:: formak.ast_tools.CompileState

   .. py:attribute:: indent
      :canonical: formak.ast_tools.CompileState.indent
      :type: int
      :value: 0

      .. autodoc2-docstring:: formak.ast_tools.CompileState.indent

.. py:class:: BaseAst()
   :canonical: formak.ast_tools.BaseAst

   Bases: :py:obj:`ast.AST`

   .. autodoc2-docstring:: formak.ast_tools.BaseAst

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.BaseAst.__init__

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.BaseAst.compile
      :abstractmethod:

      .. autodoc2-docstring:: formak.ast_tools.BaseAst.compile

   .. py:method:: indent(options: formak.ast_tools.CompileState)
      :canonical: formak.ast_tools.BaseAst.indent

      .. autodoc2-docstring:: formak.ast_tools.BaseAst.indent

.. py:function:: autoindent(compile_func)
   :canonical: formak.ast_tools.autoindent

   .. autodoc2-docstring:: formak.ast_tools.autoindent

.. py:class:: Public()
   :canonical: formak.ast_tools.Public

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Public

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Public.__init__

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Public.compile

      .. autodoc2-docstring:: formak.ast_tools.Public.compile

.. py:class:: Private()
   :canonical: formak.ast_tools.Private

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Private

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Private.__init__

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Private.compile

      .. autodoc2-docstring:: formak.ast_tools.Private.compile

.. py:class:: Arg()
   :canonical: formak.ast_tools.Arg

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Arg

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Arg.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.Arg._fields
      :value: ('type_', 'name')

      .. autodoc2-docstring:: formak.ast_tools.Arg._fields

   .. py:attribute:: type_
      :canonical: formak.ast_tools.Arg.type_
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Arg.type_

   .. py:attribute:: name
      :canonical: formak.ast_tools.Arg.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Arg.name

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Arg.compile

      .. autodoc2-docstring:: formak.ast_tools.Arg.compile

.. py:class:: Namespace()
   :canonical: formak.ast_tools.Namespace

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Namespace

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Namespace.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.Namespace._fields
      :value: ('name', 'body')

      .. autodoc2-docstring:: formak.ast_tools.Namespace._fields

   .. py:attribute:: name
      :canonical: formak.ast_tools.Namespace.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Namespace.name

   .. py:attribute:: body
      :canonical: formak.ast_tools.Namespace.body
      :type: typing.Iterable[formak.ast_tools.BaseAst]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Namespace.body

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Namespace.compile

      .. autodoc2-docstring:: formak.ast_tools.Namespace.compile

.. py:class:: HeaderFile()
   :canonical: formak.ast_tools.HeaderFile

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.HeaderFile

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.HeaderFile.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.HeaderFile._fields
      :value: ('pragma', 'includes', 'namespaces')

      .. autodoc2-docstring:: formak.ast_tools.HeaderFile._fields

   .. py:attribute:: pragma
      :canonical: formak.ast_tools.HeaderFile.pragma
      :type: bool
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.HeaderFile.pragma

   .. py:attribute:: includes
      :canonical: formak.ast_tools.HeaderFile.includes
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.HeaderFile.includes

   .. py:attribute:: namespaces
      :canonical: formak.ast_tools.HeaderFile.namespaces
      :type: typing.List[formak.ast_tools.Namespace]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.HeaderFile.namespaces

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.HeaderFile.compile

      .. autodoc2-docstring:: formak.ast_tools.HeaderFile.compile

.. py:class:: SourceFile()
   :canonical: formak.ast_tools.SourceFile

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.SourceFile

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.SourceFile.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.SourceFile._fields
      :value: ('includes', 'namespaces')

      .. autodoc2-docstring:: formak.ast_tools.SourceFile._fields

   .. py:attribute:: includes
      :canonical: formak.ast_tools.SourceFile.includes
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.SourceFile.includes

   .. py:attribute:: namespaces
      :canonical: formak.ast_tools.SourceFile.namespaces
      :type: typing.List[formak.ast_tools.Namespace]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.SourceFile.namespaces

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.SourceFile.compile

      .. autodoc2-docstring:: formak.ast_tools.SourceFile.compile

.. py:class:: ClassDef()
   :canonical: formak.ast_tools.ClassDef

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.ClassDef

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.ClassDef.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.ClassDef._fields
      :value: ('tag', 'name', 'bases', 'body')

      .. autodoc2-docstring:: formak.ast_tools.ClassDef._fields

   .. py:attribute:: tag
      :canonical: formak.ast_tools.ClassDef.tag
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ClassDef.tag

   .. py:attribute:: name
      :canonical: formak.ast_tools.ClassDef.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ClassDef.name

   .. py:attribute:: bases
      :canonical: formak.ast_tools.ClassDef.bases
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ClassDef.bases

   .. py:attribute:: body
      :canonical: formak.ast_tools.ClassDef.body
      :type: typing.Iterable[formak.ast_tools.BaseAst]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ClassDef.body

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.ClassDef.compile

      .. autodoc2-docstring:: formak.ast_tools.ClassDef.compile

.. py:class:: EnumClassDef()
   :canonical: formak.ast_tools.EnumClassDef

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.EnumClassDef

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.EnumClassDef.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.EnumClassDef._fields
      :value: ('name', 'members')

      .. autodoc2-docstring:: formak.ast_tools.EnumClassDef._fields

   .. py:attribute:: name
      :canonical: formak.ast_tools.EnumClassDef.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.EnumClassDef.name

   .. py:attribute:: members
      :canonical: formak.ast_tools.EnumClassDef.members
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.EnumClassDef.members

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.EnumClassDef.compile

      .. autodoc2-docstring:: formak.ast_tools.EnumClassDef.compile

.. py:class:: ForwardClassDeclaration()
   :canonical: formak.ast_tools.ForwardClassDeclaration

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.ForwardClassDeclaration._fields
      :value: ('tag', 'name')

      .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration._fields

   .. py:attribute:: tag
      :canonical: formak.ast_tools.ForwardClassDeclaration.tag
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration.tag

   .. py:attribute:: name
      :canonical: formak.ast_tools.ForwardClassDeclaration.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration.name

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.ForwardClassDeclaration.compile

      .. autodoc2-docstring:: formak.ast_tools.ForwardClassDeclaration.compile

.. py:class:: MemberDeclaration()
   :canonical: formak.ast_tools.MemberDeclaration

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.MemberDeclaration._fields
      :value: ('type_', 'name', 'value')

      .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration._fields

   .. py:attribute:: type_
      :canonical: formak.ast_tools.MemberDeclaration.type_
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration.type_

   .. py:attribute:: name
      :canonical: formak.ast_tools.MemberDeclaration.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration.name

   .. py:attribute:: value
      :canonical: formak.ast_tools.MemberDeclaration.value
      :type: typing.Optional[typing.Any]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration.value

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.MemberDeclaration.compile

      .. autodoc2-docstring:: formak.ast_tools.MemberDeclaration.compile

.. py:class:: UsingDeclaration()
   :canonical: formak.ast_tools.UsingDeclaration

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.UsingDeclaration._fields
      :value: ('name', 'type_')

      .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration._fields

   .. py:attribute:: name
      :canonical: formak.ast_tools.UsingDeclaration.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration.name

   .. py:attribute:: type_
      :canonical: formak.ast_tools.UsingDeclaration.type_
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration.type_

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.UsingDeclaration.compile

      .. autodoc2-docstring:: formak.ast_tools.UsingDeclaration.compile

.. py:class:: ConstructorDeclaration()
   :canonical: formak.ast_tools.ConstructorDeclaration

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.ConstructorDeclaration._fields
      :value: ('args',)

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration._fields

   .. py:attribute:: args
      :canonical: formak.ast_tools.ConstructorDeclaration.args
      :type: typing.Optional[typing.Iterable[formak.ast_tools.Arg]]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration.args

   .. py:method:: compile(options: formak.ast_tools.CompileState, classname: str, **kwargs)
      :canonical: formak.ast_tools.ConstructorDeclaration.compile

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDeclaration.compile

.. py:class:: ConstructorDefinition()
   :canonical: formak.ast_tools.ConstructorDefinition

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.ConstructorDefinition._fields
      :value: ('classname', 'args', 'initializer_list')

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition._fields

   .. py:attribute:: classname
      :canonical: formak.ast_tools.ConstructorDefinition.classname
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition.classname

   .. py:attribute:: args
      :canonical: formak.ast_tools.ConstructorDefinition.args
      :type: typing.Optional[typing.Iterable[formak.ast_tools.Arg]]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition.args

   .. py:attribute:: initializer_list
      :canonical: formak.ast_tools.ConstructorDefinition.initializer_list
      :type: typing.Optional[typing.List[typing.Tuple[str, str]]]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition.initializer_list

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.ConstructorDefinition.compile

      .. autodoc2-docstring:: formak.ast_tools.ConstructorDefinition.compile

.. py:class:: FunctionDef()
   :canonical: formak.ast_tools.FunctionDef

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.FunctionDef

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.FunctionDef.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.FunctionDef._fields
      :value: ('return_type', 'name', 'args', 'modifier', 'body')

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef._fields

   .. py:attribute:: return_type
      :canonical: formak.ast_tools.FunctionDef.return_type
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.return_type

   .. py:attribute:: name
      :canonical: formak.ast_tools.FunctionDef.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.name

   .. py:attribute:: args
      :canonical: formak.ast_tools.FunctionDef.args
      :type: typing.Iterable[formak.ast_tools.Arg]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.args

   .. py:attribute:: modifier
      :canonical: formak.ast_tools.FunctionDef.modifier
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.modifier

   .. py:attribute:: body
      :canonical: formak.ast_tools.FunctionDef.body
      :type: typing.Iterable[formak.ast_tools.BaseAst]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.body

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.FunctionDef.compile

      .. autodoc2-docstring:: formak.ast_tools.FunctionDef.compile

.. py:class:: FunctionDeclaration()
   :canonical: formak.ast_tools.FunctionDeclaration

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.FunctionDeclaration._fields
      :value: ('return_type', 'name', 'args', 'modifier')

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration._fields

   .. py:attribute:: return_type
      :canonical: formak.ast_tools.FunctionDeclaration.return_type
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.return_type

   .. py:attribute:: name
      :canonical: formak.ast_tools.FunctionDeclaration.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.name

   .. py:attribute:: args
      :canonical: formak.ast_tools.FunctionDeclaration.args
      :type: typing.Iterable[formak.ast_tools.Arg]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.args

   .. py:attribute:: modifier
      :canonical: formak.ast_tools.FunctionDeclaration.modifier
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.modifier

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.FunctionDeclaration.compile

      .. autodoc2-docstring:: formak.ast_tools.FunctionDeclaration.compile

.. py:class:: Return()
   :canonical: formak.ast_tools.Return

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Return

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Return.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.Return._fields
      :value: ('value',)

      .. autodoc2-docstring:: formak.ast_tools.Return._fields

   .. py:attribute:: value
      :canonical: formak.ast_tools.Return.value
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Return.value

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Return.compile

      .. autodoc2-docstring:: formak.ast_tools.Return.compile

.. py:class:: If()
   :canonical: formak.ast_tools.If

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.If

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.If.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.If._fields
      :value: ('test', 'body', 'orelse')

      .. autodoc2-docstring:: formak.ast_tools.If._fields

   .. py:attribute:: test
      :canonical: formak.ast_tools.If.test
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.If.test

   .. py:attribute:: body
      :canonical: formak.ast_tools.If.body
      :type: typing.Iterable[formak.ast_tools.BaseAst]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.If.body

   .. py:attribute:: orelse
      :canonical: formak.ast_tools.If.orelse
      :type: typing.List[typing.Any]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.If.orelse

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.If.compile

      .. autodoc2-docstring:: formak.ast_tools.If.compile

.. py:class:: Templated()
   :canonical: formak.ast_tools.Templated

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Templated

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Templated.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.Templated._fields
      :value: ('template_args', 'templated')

      .. autodoc2-docstring:: formak.ast_tools.Templated._fields

   .. py:attribute:: template_args
      :canonical: formak.ast_tools.Templated.template_args
      :type: typing.Iterable[formak.ast_tools.Arg]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Templated.template_args

   .. py:attribute:: templated
      :canonical: formak.ast_tools.Templated.templated
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Templated.templated

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Templated.compile

      .. autodoc2-docstring:: formak.ast_tools.Templated.compile

.. py:class:: FromFileTemplate()
   :canonical: formak.ast_tools.FromFileTemplate

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.FromFileTemplate._fields
      :value: ('name', 'inserts')

      .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate._fields

   .. py:attribute:: name
      :canonical: formak.ast_tools.FromFileTemplate.name
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate.name

   .. py:attribute:: inserts
      :canonical: formak.ast_tools.FromFileTemplate.inserts
      :type: typing.Optional[typing.Dict[str, typing.Any]]
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate.inserts

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.FromFileTemplate.compile

      .. autodoc2-docstring:: formak.ast_tools.FromFileTemplate.compile

.. py:class:: Escape()
   :canonical: formak.ast_tools.Escape

   Bases: :py:obj:`formak.ast_tools.BaseAst`

   .. autodoc2-docstring:: formak.ast_tools.Escape

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ast_tools.Escape.__init__

   .. py:attribute:: _fields
      :canonical: formak.ast_tools.Escape._fields
      :value: ('string',)

      .. autodoc2-docstring:: formak.ast_tools.Escape._fields

   .. py:attribute:: string
      :canonical: formak.ast_tools.Escape.string
      :type: str
      :value: None

      .. autodoc2-docstring:: formak.ast_tools.Escape.string

   .. py:method:: compile(options: formak.ast_tools.CompileState, **kwargs)
      :canonical: formak.ast_tools.Escape.compile

      .. autodoc2-docstring:: formak.ast_tools.Escape.compile
