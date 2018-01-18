import numpy as np
import keyword


# Python keywords
python_keywords = keyword.kwlist

# Java keywords from https://docs.oracle.com/javase/tutorial/java/nutsandbolts/_keywords.html
java_keywords = ["abstract","continue","for","new","switch","assert","default","goto","package","synchronized",
                 "boolean","do","if","private","this","break","double","implements","protected","throw",
                 "byte","else","import","public","throws","case","enum","instanceof","return","transient","catch",
                 "extends","int","short","try","char","final","interface","static","void","class","finally","long",
                 "strictfp","volatile","const","float","native","super","while"]

# Ruby keywords from http://docs.ruby-lang.org/en/2.2.0/keywords_rdoc.html
ruby_keywords = ["__ENCODING__","__LINE__","__FILE__","BEGIN","END","alias","and","begin","break",
                 "case","class","def","defined?","do","else","elsif","end","ensure","false","for","if",
                 "in","module","next","nil","not","or","redo","rescue","retry","return","self","super",
                 "then","true","undef","unless","until","when","while","yield"]

# PHP keywords form http://php.net/manual/en/reserved.keywords.php
php_keywords = ["__halt_compiler","abstract","and","array","as","break","callable","case","catch","class","clone",
                "const","continue","declare","default","die","do","echo","else","elseif","empty","enddeclare",
                "endfor","endforeach","endif","endswitch","endwhile","eval","exit","extends","final","finally",
                "for","foreach","function","global","goto","if","implements","include","include_once","instanceof",
                "insteadof","interface","isset","list","namespace","new","or","print","private","protected",
                "public","require","require_once","return","static","switch","throw","trait","try","unset","use",
                "var","while","xor","yield"]

cpp_keywords = ["auto","const", "double", "float", "int", "short", "struct", "unsigned", "break", "continue",
                "else", "for", "long", "signed", "switch", "void", "case", "default", "enum", "goto", "register",
                "sizeof", "typedef", "volatile", "char", "do", "extern", "if", "return", "static", "union", "while",
                "asm", "dynamic_cast", "namespace", "reinterpret_cast", "try", "bool", "explicit", "new",
                "static_cast", "typeid", "catch", "false", "operator", "template", "typename", "class", "friend",
                "private", "this", "using", "const_cast", "inline", "public", "throw", "virtual", "delete", "mutable",
                "protected", "true", "wchar_t", "and", "bitand", "compl", "not_eq", "or_eq", "xor_eq", "and_eq",
                "bitor", "not", "or", "xor"]

c_sharp_keywords = ["abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked", "class",
                    "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event",
                    "explicit", "extern", "false", "finally", "fixed", "float", "for", "foreach", "goto", "if",
                    "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace", "new",
                    "null", "object", "operator", "out", "override", "params", "private", "protected", "public",
                    "readonly", "ref", "return", "sbyte", "sealed", "short", "sizeof", "stackalloc", "static",
                    "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong",
                    "unchecked", "unsafe", "ushort", "using", "virtual", "void", "volatile", "while"]

javascript_keywords = ["abstract", "arguments", "boolean", "break", "byte", "case", "catch", "char", "class",
                       "const", "continue", "debugger", "default", "delete", "do", "double", "else", "enum",
                       "eval", "export", "extends", "false", "final", "finally", "float", "for", "function",
                       "goto", "if", "implements", "import", "in", "instanceof", "int", "interface", "let",
                       "long", "native", "new", "null", "package", "private", "protected", "public", "return",
                       "short", "static", "super", "switch", "synchronized", "this", "throw", "throws",
                       "transient", "true", "try", "typeof", "var", "void", "volatile", "while", "with", "yield"]

coffeescript_keywords = ["case", "default", "function", "var", "void", "with", "const", "let", "enum", "export",
                         "import", "native", "__hasProp", "__extends", "__slice", "__bind", "__indexOf", "implements",
                         "interface", "package", "private", "protected", "public", "static", "yield", "true", "false",
                         "null", "this", "new", "delete", "typeof", "in", "arguments", "eval", "instanceof", "return",
                         "throw", "break", "continue", "debugger", "if", "else", "switch", "for", "while", "do", "try",
                         "catch", "finally", "class", "extends", "super", "undefined", "then", "unless", "until",
                         "loop", "of", "by", "when", "and", "or", "is", "isnt", "not", "yes", "no", "on", "off"]

R_keywords = ["if", "else", "repeat", "while", "function", "for in", "next", "break", "TRUE", "FALSE", "NULL",
              "Inf", "NaN", "NA", "NA_integer_", "NA_real_", "NA_complex_", "NA_character_"]

typeScript_keywords = ["break", "as", "any", "case", "implements", "boolean", "catch", "interface", "constructor",
                       "class", "let", "declare", "const", "package", "get", "continue", "private", "module",
                       "debugger", "protected", "require", "default", "public", "number", "delete", "static",
                       "set", "do", "yield", "string", "else", "symbol", "enum", "type", "export", "from",
                       "extends", "of", "false", "finally", "for", "function", "if", "import", "in", "instanceof",
                       "new", "null", "return", "super", "switch", "this", "throw", "true", "try", "typeof",
                       "var", "void", "while", "with"]
hashkell_keywords =  ["case","class","data","default","deriving","do","else","forall","if","import","in","infix",
                      "infixl","infixr","instance","let","module","newtype","of","qualified","then",
                      "type","where","foreign","ccall","as","safe","unsafe"]

all_keywords = python_keywords + java_keywords + ruby_keywords + php_keywords + cpp_keywords \
                + c_sharp_keywords + javascript_keywords + coffeescript_keywords + R_keywords \
                + typeScript_keywords + hashkell_keywords


all_keywords = np.unique(all_keywords).tolist()


tag_names = ["2016-06","2016-01","2015-06","2015-01","2014-06","2014-01","2013-06","2013-01",
"2012-06","2012-01","2011-06","2011-01","2010-06","2010-01","2009-06","2009-01",
"2008-06","2008-01","2007-06","2007-01","2006-06","2006-01","2005-06","2005-01",
"2004-06","2004-01","2003-06","2003-01"]

# --> Not used???
project_type_map = {
	"androidannotations-tags": ".java",
	"bigbluebutton-tags": ".java",
	"cassandra-tags": ".java",
	"elasticsearch-tags": ".java",
	"hibernate-orm-tags": ".java",
	"liferay-portal-tags": ".java",
	"netty-tags": ".java",
	"platform_frameworks_base-tags": ".java",
	"spring-framework-tags": ".java",
	"wildfly-tags": ".java",
	"laravel-tags": ".php",
	"symfony-tags": ".php",
	"cakephp-tags": ".php",
	"CodeIgniter-tags": ".php",
	"rails-tags": ".rb",
	"sinatra-tags": ".rb",
	"padrino-framework-tags": ".rb",
	"hanami-tags": ".rb",
	"pakyow-tags": ".rb",
	"flask-tags": ".py",
	"django-tags": ".py",
	"web2py-tags": ".py",
	"frappe-tags": ".py",
	"ninja-tags": ".java",
	"meteor-tags": "javascript",
	"express-tags": "javascript",
	"sails-tags": "javascript",
	"mean-tags": "javascript",
	"derby-tags": "javascript",
	"nodal-tags": "javascript"
}
