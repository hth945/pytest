测试功能
=========================
点灯机

* This is a bulleted list.
* It has two items, the second
  item uses two lines.

1. This is a numbered list.
2. It has two items too.

#. This is a numbered list.
#. It has two items too.

* this is
* a list

  * with a nested list
  * and some subitems
    * with a nested list

* and here the parent list continues
  term (up to a line of text)
  Definition of the term, which must be indented

  and can even consist of multiple paragraphs

  next term
  Description.

| These lines are
| broken exactly like in
| the source file.

This is a normal text paragraph. The next paragraph is a code sample::

   It is not processed in any way, except
   that the indentation is removed.

   It can span multiple lines.

This is a normal text paragraph again.

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | ...        | ...      |          |
+------------------------+------------+----------+----------+

=====  =====  =======
A      B      A and B
=====  =====  =======
False  False  False
True   False  False
False  True   False
True   True   True
=====  =====  =======

This is a paragraph that contains `a link`_.

.. _a link: http://example.com/

..
   This whole indented block
   is a comment.

   Still in the comment.

   
=================
This is a heading
=================

=================
This is a hading2
=================


=================
This is aheading3
=================



.. |name| replace:: replacement *text*

.. |caution| image:: 1.bmp
             :alt: Warning!

.. function:: foo(x)
              foo(y, z)
   :module: some.module.name

   Return a line of text input from the user.

.. image:: 1.bmp

Lorem ipsum [#f1]_ dolor sit amet ... [#f2]_

.. rubric:: Footnotes

.. [#f1] Text of the first footnote.
.. [#f2] Text of the second footnote.


Lorem ipsum [Ref]_ dolor sit amet.

.. [Ref] Book or article reference, URL or whatever.
