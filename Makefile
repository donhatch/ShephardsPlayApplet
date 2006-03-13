JAVAROOT=/usr/java/jdk1.5.0
JAVAC=${JAVAROOT}/bin/javac

#JAVAROOT=/usr/java/j2sdk1.4.2
#JAVAC=jikes +P -source 1.4 -classpath ${JAVAROOT}/jre/lib/rt.jar

JARFILE = ShephardsPlayApplet.jar
CLASSES = \
        GraphicsAntiAliasingSetter.class \
        MyGraphics.class \
        ShephardsPlayApplet.class \
        ${NULL}
JAR_DEPENDS_ON = ${CLASSES}      macros.h Makefile javacpp javarenumber
JAR_CONTAINS = *.class *.prejava macros.h Makefile javacpp javarenumber

.PHONY: all
all: ${JARFILE}

${JARFILE}: Makefile META-INF/MANIFEST.MF ${CLASSES}
	${JAVAROOT}/bin/jar -cfm ${JARFILE} META-INF/MANIFEST.MF ${JAR_CONTAINS}

CPPFLAGS += -Wall -Werror
# The following seems to work but clutters up output and may be less portable
#CPPFLAGS += -pedantic -std=c99
# The following is too strict for me (requires #'s to be non-indented)
#CPPFLAGS += -Wtraditional

.SUFFIXES: .prejava .java .class
.prejava.class:
	./javacpp ${CPPFLAGS} ${JAVAC} -classpath ".:./donhatchsw.jar" $*.prejava
	./javarenumber -v 0 $*.class
	# too slow... only do this in the production version
	# on second thought, try it, for now...
	# on third hand, it bombs with Couldn't open GraphicsAntiAliasingSetter$*.class because that one has no subclasses... argh.
	#@./javarenumber -v -1 $*'$$'*.class

MyGraphics.class: macros.h Makefile
GraphicsAntiAliasingSetter.class: macros.h Makefile
ShephardsPlayApplet.class: macros.h Makefile

.PHONY: clean
clean:
	rm -f core ${JARFILE} *.class *.java.lines *.java

